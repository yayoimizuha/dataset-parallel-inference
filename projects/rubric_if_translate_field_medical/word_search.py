"""
Dense-vector embedding pipeline for medical term search.

Architecture (streaming overlap)
=================================
Producer (background thread + multiprocessing.Pool):
    Wikipedia dataset iteration
    -> multiprocessing.Pool.imap for CPU-parallel semchunk splitting
    -> yields encode-ready batches into asyncio.Queue

Consumer (asyncio event loop, 8 GPUs):
    Pulls batches from asyncio.Queue
    -> tokenize -> round-robin dispatch to 8 ONNX Runtime GPU sessions
    -> collect embeddings + metadata

Output:
    polars DataFrame -> zstd-compressed parquet
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl
import semchunk
import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# ======================================================================
# 定数
# ======================================================================

_CHUNK_SIZE = 8192
_UUID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

_ENCODE_BATCH_SIZE = 16  # ONNX 推論 1 回あたりのバッチサイズ (8192 トークン長のため控えめに)
_QUEUE_MAXSIZE = 32  # producer → consumer 間のバッファ上限
_CONCURRENT_PER_GPU = 3  # 1 GPU あたりの同時推論バッチ数

_NUM_GPUS = 8
_NUM_PROC = int(max(4, (os.cpu_count() or 1) / 2 - 10))  # semchunk 並列ワーカー数

_ONNX_REPO = "sirasagi62/ruri-v3-310m-ONNX"
_ONNX_FILENAME = "onnx/model_fp16.onnx"
_TOKENIZER_REPO = "sirasagi62/ruri-v3-310m-ONNX"

_DATASET_NAME = "omarkamali/wikipedia-monthly"
_DATASET_CONFIG = "latest.en"

_OUTPUT_DIR = Path(__file__).parent / "output"
_OUTPUT_FILENAME = "medical_terms_embeddings.parquet"

# センチネル: Queue の終端を示す
_SENTINEL = None


# ======================================================================
# データ構造
# ======================================================================


@dataclass
class ChunkBatch:
    """Producer から Consumer へ渡すバッチ単位のデータ。"""

    point_ids: list[str]
    article_ids: list[str]
    titles: list[str]
    chunk_texts: list[str]
    chunk_indices: list[int]


@dataclass
class ResultAccumulator:
    """Consumer 側で embedding とメタデータを蓄積する。"""

    point_ids: list[str] = field(default_factory=list)
    article_ids: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    chunk_texts: list[str] = field(default_factory=list)
    chunk_indices: list[int] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)

    def extend(self, batch: ChunkBatch, emb: np.ndarray) -> None:
        self.point_ids.extend(batch.point_ids)
        self.article_ids.extend(batch.article_ids)
        self.titles.extend(batch.titles)
        self.chunk_texts.extend(batch.chunk_texts)
        self.chunk_indices.extend(batch.chunk_indices)
        self.embeddings.append(emb)


# ======================================================================
# ユーティリティ (numpy)
# ======================================================================


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row vector."""
    norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


# ======================================================================
# Producer: Dataset loading + CPU parallel chunking → asyncio.Queue
# ======================================================================

# multiprocessing ワーカーごとの chunker キャッシュ
_chunker_cache = None


def _get_chunker():
    global _chunker_cache
    if _chunker_cache is None:
        tok = AutoTokenizer.from_pretrained(_TOKENIZER_REPO)
        # noinspection PyTypeChecker
        _chunker_cache = semchunk.chunkerify(tok, chunk_size=_CHUNK_SIZE)
    return _chunker_cache


def _chunk_one_article(article: dict) -> list[dict]:
    """
    1 記事をチャンク分割する。multiprocessing.Pool のワーカー関数。

    Returns:
        チャンクごとの dict のリスト。
    """
    chunker = _get_chunker()
    data_id = article["id"]
    title = article["title"]
    text = article["text"]

    results = []
    for ci, chunk in enumerate(chunker(text)):
        results.append({
            "point_id": str(uuid.uuid5(_UUID_NAMESPACE, f"{data_id}_{ci}")),
            "article_id": data_id,
            "title": title,
            "chunk_text": chunk,
            "chunk_index": ci,
        })
    return results


def _producer_thread(
        queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        total_articles: int | None,
) -> None:
    """
    バックグラウンドスレッドで実行。
    multiprocessing.Pool で記事を並列チャンク分割し、
    _ENCODE_BATCH_SIZE 件ごとに ChunkBatch として asyncio.Queue に投入する。
    """
    wiki = load_dataset(_DATASET_NAME, _DATASET_CONFIG, split="train", streaming=False)
    print(f"[Producer] Dataset loaded: {len(wiki)} articles", flush=True)

    batch_buf: list[dict] = []

    def _flush_batch() -> None:
        nonlocal batch_buf
        if not batch_buf:
            return
        cb = ChunkBatch(
            point_ids=[d["point_id"] for d in batch_buf],
            article_ids=[d["article_id"] for d in batch_buf],
            titles=[d["title"] for d in batch_buf],
            chunk_texts=[d["chunk_text"] for d in batch_buf],
            chunk_indices=[d["chunk_index"] for d in batch_buf],
        )
        # スレッドセーフに asyncio.Queue へ put
        future = asyncio.run_coroutine_threadsafe(queue.put(cb), loop)
        future.result()  # バックプレッシャ: Queue が満杯なら待つ
        batch_buf = []

    progress = tqdm.tqdm(total=total_articles, desc="[Producer] Chunking")

    with mp.Pool(processes=_NUM_PROC) as pool:
        for chunks in pool.imap_unordered(_chunk_one_article, wiki, chunksize=64):
            batch_buf.extend(chunks)
            progress.update(1)
            while len(batch_buf) >= _ENCODE_BATCH_SIZE:
                to_send = batch_buf[:_ENCODE_BATCH_SIZE]
                batch_buf = batch_buf[_ENCODE_BATCH_SIZE:]
                cb = ChunkBatch(
                    point_ids=[d["point_id"] for d in to_send],
                    article_ids=[d["article_id"] for d in to_send],
                    titles=[d["title"] for d in to_send],
                    chunk_texts=[d["chunk_text"] for d in to_send],
                    chunk_indices=[d["chunk_index"] for d in to_send],
                )
                future = asyncio.run_coroutine_threadsafe(queue.put(cb), loop)
                future.result()

    # 残余フラッシュ
    _flush_batch()
    progress.close()

    # 終端シグナル
    future = asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop)
    future.result()
    print("[Producer] Done.", flush=True)


# ======================================================================
# ONNX Runtime session management
# ======================================================================


def _download_onnx_model() -> str:
    """HuggingFace Hub から ONNX モデルをダウンロードし、ローカルパスを返す。"""
    print(f"[Setup] Downloading ONNX model: {_ONNX_REPO}/{_ONNX_FILENAME}", flush=True)
    local_path = hf_hub_download(repo_id=_ONNX_REPO, filename=_ONNX_FILENAME)
    print(f"[Setup] Model cached at: {local_path}", flush=True)
    return local_path


def _create_onnx_sessions(model_path: str, num_gpus: int) -> list[ort.InferenceSession]:
    """各 GPU に ONNX Runtime セッションを 1 つずつ作成する。"""
    sessions: list[ort.InferenceSession] = []
    threads_per_gpu = max(1, (os.cpu_count() or 8) // num_gpus)

    for gpu_id in range(num_gpus):
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = threads_per_gpu
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=[
                ("CUDAExecutionProvider", {"device_id": str(gpu_id)}),
                "CPUExecutionProvider",
            ],
        )
        sessions.append(session)
        print(f"[Setup] ONNX session created on GPU {gpu_id}", flush=True)

    return sessions


# ======================================================================
# Consumer: async GPU inference
# ======================================================================


def _run_inference(
        session: ort.InferenceSession,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
) -> np.ndarray:
    """
    1 バッチの ONNX 推論を実行し、L2 正規化済み dense embedding を返す。
    ThreadPoolExecutor 内から呼び出される (GIL は ORT の C++ 層で解放)。

    ONNX モデル出力:
        - token_embeddings: (batch, seq_len, 768) — 各トークンの隠れ状態
        - sentence_embedding: (batch, 768) — pooling 済み文ベクトル
    """
    outputs = session.run(
        ["sentence_embedding"],
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )
    sentence_embedding = outputs[0]  # (batch, 768), float32
    return _l2_normalize(sentence_embedding)


async def _consumer(
        queue: asyncio.Queue,
        sessions: list[ort.InferenceSession],
        tokenizer: AutoTokenizer,
        result: ResultAccumulator,
) -> None:
    """
    asyncio.Queue からバッチを取り出し、8 GPU にラウンドロビンで推論を投入する。
    GPU ごとにセマフォで排他制御し、各 GPU が常に最大 _CONCURRENT_PER_GPU バッチ
    処理中になるようパイプラインを飽和させる。
    """
    num_gpus = len(sessions)
    executor = ThreadPoolExecutor(max_workers=num_gpus * _CONCURRENT_PER_GPU)
    gpu_sems = [asyncio.Semaphore(_CONCURRENT_PER_GPU) for _ in range(num_gpus)]
    batch_counter = 0
    progress = tqdm.tqdm(desc="[Consumer] Encoding", unit="batch")

    async def _process_one(batch: ChunkBatch, gpu_id: int) -> None:
        prefixed = [f"検索文書: {t}" for t in batch.chunk_texts]
        encoded = tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        loop = asyncio.get_event_loop()
        async with gpu_sems[gpu_id]:
            emb = await loop.run_in_executor(
                executor, _run_inference, sessions[gpu_id], input_ids, attention_mask,
            )

        result.extend(batch, emb)
        progress.update(1)

    pending_tasks: list[asyncio.Task] = []

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break

        gpu_id = batch_counter % num_gpus
        batch_counter += 1
        task = asyncio.create_task(_process_one(item, gpu_id))
        pending_tasks.append(task)

    # 残りの推論タスクを待機
    if pending_tasks:
        await asyncio.gather(*pending_tasks)

    progress.close()
    executor.shutdown(wait=True)
    print(f"[Consumer] Done. {batch_counter} batches processed.", flush=True)


# ======================================================================
# Parquet output
# ======================================================================


def _save_to_parquet(result: ResultAccumulator, output_path: Path) -> None:
    """蓄積した結果を polars DataFrame にまとめて parquet で保存する。"""
    total = len(result.point_ids)
    print(f"[Output] Building polars DataFrame ({total} rows)...", flush=True)

    all_embeddings = np.concatenate(result.embeddings, axis=0)

    df = pl.DataFrame(
        {
            "point_id": result.point_ids,
            "article_id": result.article_ids,
            "title": result.titles,
            "chunk_text": result.chunk_texts,
            "chunk_index": pl.Series(result.chunk_indices, dtype=pl.UInt32),
            "embedding": [row.tolist() for row in all_embeddings],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="zstd")
    size_gb = output_path.stat().st_size / 1e9
    print(f"[Output] Saved to {output_path} ({size_gb:.2f} GB, {total} rows)", flush=True)


# ======================================================================
# エントリーポイント
# ======================================================================


async def _async_main() -> None:
    # ------------------------------------------------------------------
    # Setup: ONNX model + sessions + tokenizer
    # ------------------------------------------------------------------
    model_path = _download_onnx_model()
    sessions = _create_onnx_sessions(model_path, _NUM_GPUS)
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_REPO)
    tokenizer.model_max_length = 8192

    # ------------------------------------------------------------------
    # 記事数を先に取得 (progress bar 用)
    # ------------------------------------------------------------------
    wiki_meta = load_dataset(_DATASET_NAME, _DATASET_CONFIG, split="train", streaming=False)
    total_articles = len(wiki_meta)
    del wiki_meta
    print(f"[Main] Total articles: {total_articles}", flush=True)

    # ------------------------------------------------------------------
    # Producer → Consumer をオーバーラップ実行
    # ------------------------------------------------------------------
    queue: asyncio.Queue[ChunkBatch | None] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    result = ResultAccumulator()

    loop = asyncio.get_event_loop()

    # Producer をバックグラウンドスレッドで起動
    producer_future = loop.run_in_executor(
        None, _producer_thread, queue, loop, total_articles,
    )

    # Consumer は asyncio タスクとして実行
    await _consumer(queue, sessions, tokenizer, result)

    # Producer スレッドの終了を待機
    await producer_future

    del sessions
    print(f"[Main] All encoding done. Total chunks: {len(result.point_ids)}", flush=True)

    # ------------------------------------------------------------------
    # Parquet 出力
    # ------------------------------------------------------------------
    _save_to_parquet(result, _OUTPUT_DIR / _OUTPUT_FILENAME)
    print("Pipeline completed.", flush=True)


def main() -> None:
    """同期エントリーポイント。"""
    mp.set_start_method("spawn", force=True)
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
