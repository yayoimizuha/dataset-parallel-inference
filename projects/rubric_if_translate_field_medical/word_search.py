"""
4-process pipeline for medical term registration into Qdrant.

Process 1 (dataset_loader): Dataset read + chunking + duplicate check -> sends to P2, P3, P4
Process 2 (dense_encoder):  Dense vector generation on cuda:0 -> sends to P4
Process 3 (sparse_encoder): Sparse vector generation on cuda:1 -> sends to P4
Process 4 (qdrant_writer):  Merges dense+sparse+meta results, upserts to Qdrant
"""
import json
import sys
import threading
import traceback
import uuid
from typing import Any

import numpy as np
import semchunk
import torch
import torch.multiprocessing as mp
# noinspection PyPep8Naming
import torch.nn.functional as F
import tqdm
import yasem
from datasets import load_dataset
from qdrant_client import QdrantClient, models
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer, BitsAndBytesConfig

# ======================================================================
# 定数
# ======================================================================

_CHUNK_SIZE = 512
_UUID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

_DENSE_BATCH_SIZE = 4096
_SPARSE_BATCH_SIZE = 4096
_UPSERT_BATCH_SIZE = 500
_LOADER_BATCH_SIZE = 2048

_QDRANT_URL = "http://localhost:6333"
_COLLECTION_NAME = "medical_terms"

_DENSE_MODEL = "cl-nagoya/ruri-v3-310m"
_SPARSE_MODEL = "hotchpotch/japanese-splade-v2"
_DATASET_NAME = "omarkamali/wikipedia-monthly"
_DATASET_CONFIG = "latest.en"


class _Sentinel:
    """pickle 可能なパイプライン終了シグナル。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __reduce__(self):
        return (self.__class__, ())


SENTINEL = _Sentinel()


# ======================================================================
# ユーティリティ
# ======================================================================

def _mean_pooling(model_output, attention_mask) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (token_embeddings * mask_expanded).sum(1)
    sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def _ensure_collection(qdrant: QdrantClient) -> None:
    """コレクションが存在しなければ作成する。複数プロセスからの同時呼び出しに対応。"""
    if qdrant.collection_exists(_COLLECTION_NAME):
        return
    try:
        qdrant.create_collection(
            _COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            sparse_vectors_config={
                "splade": models.SparseVectorParams(
                    index=models.SparseIndexParams(full_scan_threshold=1000)
                )
            },
        )
    except Exception:
        # 別プロセスが先に作成した場合は無視
        if not qdrant.collection_exists(_COLLECTION_NAME):
            raise


def _send_batch(batch_id: int, batch_buf: list[dict[str, Any]], *queues: mp.Queue) -> None:
    """同一バッチを複数の Queue へ送信する。"""
    msg = (batch_id, batch_buf)
    for q in queues:
        q.put(msg)


# ======================================================================
# Process 1: Dataset loader
# ======================================================================

def dataset_loader(
    to_dense_q: mp.Queue,
    to_sparse_q: mp.Queue,
    to_writer_q: mp.Queue,
) -> None:
    """
    Wikipedia データセットを読み込み、チャンク分割・未登録チェックを行い、
    バッチ単位で P2 (dense), P3 (sparse), P4 (メタデータ) へ送信する。
    """
    try:
        print("[P1] Starting dataset loader...", flush=True)

        qdrant = QdrantClient(url=_QDRANT_URL)
        _ensure_collection(qdrant)

        wiki_dataset = load_dataset(_DATASET_NAME, _DATASET_CONFIG, split="train", streaming=False)
        splade_tokenizer = AutoTokenizer.from_pretrained(_SPARSE_MODEL)
        # noinspection PyTypeChecker
        chunker = semchunk.chunkerify(splade_tokenizer, chunk_size=_CHUNK_SIZE)

        total = wiki_dataset.info.splits["train"].num_examples  # type: ignore[union-attr]
        batch_buf: list[dict[str, Any]] = []
        batch_id = 0
        sent_chunks = 0

        _item: dict
        for _item in tqdm.tqdm(wiki_dataset, total=total, desc="[P1] Chunking"):  # type: ignore[assignment]
            _data_id: str = _item["id"]
            _title: str = _item["title"]
            _text: str = _item["text"]

            _chunks: list[str] = chunker(_text)
            _point_ids = [
                str(uuid.uuid5(_UUID_NAMESPACE, f"{_data_id}_{ci}"))
                for ci in range(len(_chunks))
            ]

            _existing = qdrant.retrieve(
                _COLLECTION_NAME, ids=_point_ids, with_payload=False, with_vectors=False,
            )
            _existing_ids = {p.id for p in _existing}

            for ci, (_chunk, _pid) in enumerate(zip(_chunks, _point_ids)):
                if _pid in _existing_ids:
                    continue
                batch_buf.append({
                    "point_id": _pid,
                    "article_id": _data_id,
                    "title": _title,
                    "chunk_text": _chunk,
                    "chunk_index": ci,
                })
                if len(batch_buf) >= _LOADER_BATCH_SIZE:
                    _send_batch(batch_id, batch_buf, to_dense_q, to_sparse_q, to_writer_q)
                    sent_chunks += len(batch_buf)
                    print(f"[P1] Sent batch {batch_id} ({len(batch_buf)} chunks, total {sent_chunks})", flush=True)
                    batch_id += 1
                    batch_buf = []

        # 残りを送信
        if batch_buf:
            _send_batch(batch_id, batch_buf, to_dense_q, to_sparse_q, to_writer_q)
            sent_chunks += len(batch_buf)
            print(f"[P1] Sent final batch {batch_id} ({len(batch_buf)} chunks, total {sent_chunks})", flush=True)

        # 終了シグナル
        for q in (to_dense_q, to_sparse_q, to_writer_q):
            q.put(SENTINEL)

        qdrant.close()
        print(f"[P1] Done. Total {sent_chunks} chunks sent.", flush=True)

    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        # 終了シグナルを送って他プロセスがハングしないようにする
        for q in (to_dense_q, to_sparse_q, to_writer_q):
            try:
                q.put(SENTINEL)
            except Exception:
                pass
        raise


# ======================================================================
# Process 2: Dense encoder (cuda:0)
# ======================================================================

def dense_encoder(in_q: mp.Queue, out_q: mp.Queue) -> None:
    """ruri-v3-310m で dense ベクトルを生成し、結果を P4 へ送る。"""
    try:
        print("[P2] Starting dense encoder on cuda:0...", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(_DENSE_MODEL)
        model = AutoModel.from_pretrained(
            _DENSE_MODEL,
            # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            trust_remote_code=True,
            device_map="cuda:0",
        ).to(torch.bfloat16)
        print("[P2] Model loaded.", flush=True)

        def encode(texts: list[str]) -> np.ndarray:
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=8192, return_tensors="pt",
            )
            if hasattr(model, "device"):
                encoded = {k: v.to(model.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = model(**encoded)
            emb = _mean_pooling(output, encoded["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            return emb.cpu().float().numpy()

        while True:
            item = in_q.get()
            if isinstance(item, _Sentinel):
                out_q.put(SENTINEL)
                break

            batch_id, chunk_dicts = item
            texts = [f"検索文書: {d['chunk_text']}" for d in chunk_dicts]

            all_embs = [
                encode(texts[i: i + _DENSE_BATCH_SIZE])
                for i in range(0, len(texts), _DENSE_BATCH_SIZE)
            ]
            out_q.put((batch_id, np.concatenate(all_embs, axis=0)))
            print(f"[P2] Dense batch {batch_id} done ({len(texts)} chunks)", flush=True)

        torch.cuda.empty_cache()
        print("[P2] Done.", flush=True)

    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        # P4 がハングしないように終了シグナルを送る
        try:
            out_q.put(SENTINEL)
        except Exception:
            pass
        raise


# ======================================================================
# Process 3: Sparse encoder (cuda:1)
# ======================================================================

def sparse_encoder(in_q: mp.Queue, out_q: mp.Queue) -> None:
    """japanese-splade-v2 で sparse ベクトルを生成し、結果を P4 へ送る。"""
    try:
        print("[P3] Starting sparse encoder on cuda:1...", flush=True)

        splade_model = AutoModelForMaskedLM.from_pretrained(
            _SPARSE_MODEL,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            trust_remote_code=True,
            device_map="cuda:1",
        )
        # noinspection PyTypeChecker
        splade_embedder = yasem.SpladeEmbedder(_SPARSE_MODEL, device="cuda:1")
        splade_embedder.model = splade_model
        print("[P3] Model loaded.", flush=True)

        def encode(texts: list[str]) -> list[dict]:
            sparse_matrix = splade_embedder.encode(texts, convert_to_csr_matrix=True)
            return [
                {"indices": sparse_matrix.getrow(i).indices.tolist(),
                 "values": sparse_matrix.getrow(i).data.tolist()}
                for i in range(sparse_matrix.shape[0])
            ]

        while True:
            item = in_q.get()
            if isinstance(item, _Sentinel):
                out_q.put(SENTINEL)
                break

            batch_id, chunk_dicts = item
            texts = [d["chunk_text"] for d in chunk_dicts]

            all_sparse: list[dict] = []
            for i in range(0, len(texts), _SPARSE_BATCH_SIZE):
                all_sparse.extend(encode(texts[i: i + _SPARSE_BATCH_SIZE]))

            out_q.put((batch_id, all_sparse))
            print(f"[P3] Sparse batch {batch_id} done ({len(texts)} chunks)", flush=True)

        torch.cuda.empty_cache()
        print("[P3] Done.", flush=True)

    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        try:
            out_q.put(SENTINEL)
        except Exception:
            pass
        raise


# ======================================================================
# Process 4: Qdrant writer
# ======================================================================

_REQUIRED_KEYS = frozenset({"meta", "dense", "sparse"})


def qdrant_writer(
    from_loader_q: mp.Queue,
    from_dense_q: mp.Queue,
    from_sparse_q: mp.Queue,
) -> None:
    """
    P1 からメタデータ、P2 から dense、P3 から sparse を受け取り、
    同じ batch_id のデータが揃ったら Qdrant へ upsert する。
    """
    try:
        print("[P4] Starting Qdrant writer...", flush=True)

        qdrant = QdrantClient(url=_QDRANT_URL)
        _ensure_collection(qdrant)

        pending: dict[int, dict[str, Any]] = {}
        finished_sources = 0
        lock = threading.Lock()
        upsert_event = threading.Event()
        done_event = threading.Event()

        # -- 汎用レシーバー --------------------------------------------

        def _receiver(queue: mp.Queue, key: str) -> None:
            """Queue から受信し pending[batch_id][key] に格納する。"""
            nonlocal finished_sources
            while True:
                item = queue.get()
                if isinstance(item, _Sentinel):
                    with lock:
                        finished_sources += 1
                        if finished_sources >= 3:
                            done_event.set()
                    upsert_event.set()
                    return
                batch_id, data = item
                with lock:
                    pending.setdefault(batch_id, {})[key] = data
                upsert_event.set()

        # -- upsert ----------------------------------------------------

        def _do_upsert(bid: int) -> None:
            entry = pending.get(bid)
            if entry is None or not _REQUIRED_KEYS <= entry.keys():
                return

            chunk_dicts = entry["meta"]
            dense_vecs = entry["dense"]
            sparse_dicts = entry["sparse"]

            for i in range(0, len(chunk_dicts), _UPSERT_BATCH_SIZE):
                items = chunk_dicts[i: i + _UPSERT_BATCH_SIZE]
                d_vecs = dense_vecs[i: i + _UPSERT_BATCH_SIZE]
                s_vecs = sparse_dicts[i: i + _UPSERT_BATCH_SIZE]

                points = [
                    models.PointStruct(
                        id=meta["point_id"],
                        vector={
                            "": dense.tolist(),
                            "splade": models.SparseVector(
                                indices=sparse["indices"], values=sparse["values"],
                            ),
                        },
                        payload={
                            "article_id": meta["article_id"],
                            "title": meta["title"],
                            "chunk_text": meta["chunk_text"],
                            "chunk_index": meta["chunk_index"],
                        },
                    )
                    for meta, dense, sparse in zip(items, d_vecs, s_vecs)
                ]
                qdrant.upsert(_COLLECTION_NAME, points=points)

            print(f"[P4] Upserted batch {bid} ({len(chunk_dicts)} points)", flush=True)
            del pending[bid]

        # -- 3 つの Queue を並行受信するスレッドを起動 -----------------

        threads = [
            threading.Thread(target=_receiver, args=(from_loader_q, "meta"), daemon=True),
            threading.Thread(target=_receiver, args=(from_dense_q, "dense"), daemon=True),
            threading.Thread(target=_receiver, args=(from_sparse_q, "sparse"), daemon=True),
        ]
        for t in threads:
            t.start()

        # -- メインスレッドで upsert を処理 ----------------------------

        while True:
            upsert_event.wait(timeout=1.0)
            upsert_event.clear()

            with lock:
                ready = [bid for bid, e in pending.items() if _REQUIRED_KEYS <= e.keys()]
            for bid in sorted(ready):
                _do_upsert(bid)

            if done_event.is_set():
                with lock:
                    remaining = [bid for bid, e in pending.items() if _REQUIRED_KEYS <= e.keys()]
                for bid in sorted(remaining):
                    _do_upsert(bid)
                break

        for t in threads:
            t.join()
        qdrant.close()
        print("[P4] Done.", flush=True)

    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        raise


# ======================================================================
# パイプラインオーケストレーション
# ======================================================================

def run_parallel_registration() -> None:
    """
    4 プロセスをパイプラインで並列実行し、Qdrant に登録する。

    P1 --batch--> P2 (cuda:0, dense) --result--> P4 (Qdrant upsert)
    P1 --batch--> P3 (cuda:1, sparse) --result--> P4
    P1 --meta---> P4
    """
    mp.set_start_method("spawn", force=True)

    loader_to_dense: mp.Queue = mp.Queue(maxsize=8)
    loader_to_sparse: mp.Queue = mp.Queue(maxsize=8)
    loader_to_writer: mp.Queue = mp.Queue(maxsize=8)
    dense_to_writer: mp.Queue = mp.Queue(maxsize=8)
    sparse_to_writer: mp.Queue = mp.Queue(maxsize=8)

    processes = [
        mp.Process(target=dataset_loader,
                   args=(loader_to_dense, loader_to_sparse, loader_to_writer),
                   name="P1-Loader"),
        mp.Process(target=dense_encoder,
                   args=(loader_to_dense, dense_to_writer),
                   name="P2-Dense"),
        mp.Process(target=sparse_encoder,
                   args=(loader_to_sparse, sparse_to_writer),
                   name="P3-Sparse"),
        mp.Process(target=qdrant_writer,
                   args=(loader_to_writer, dense_to_writer, sparse_to_writer),
                   name="P4-Qdrant"),
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # 異常終了したプロセスを報告
    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        names = ", ".join(f"{p.name} (exit={p.exitcode})" for p in failed)
        raise RuntimeError(f"Processes failed: {names}")

    print("All processes finished.")


# ======================================================================
# 検索 (単一プロセス・既存互換)
# ======================================================================

class MedicalTermSearcher:
    """検索用クラス。登録は run_parallel_registration() で行う。"""

    def __init__(self):
        self.qdrant_db = QdrantClient(url=_QDRANT_URL)
        _ensure_collection(self.qdrant_db)

        # noinspection PyNoneFunctionAssignment
        self.ruri_tokenizer = AutoTokenizer.from_pretrained(_DENSE_MODEL)
        self.embedding = AutoModel.from_pretrained(
            _DENSE_MODEL,
            # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            trust_remote_code=True,
            device_map="cuda:0",
        ).to(torch.bfloat16)

        self.splade = AutoModelForMaskedLM.from_pretrained(
            _SPARSE_MODEL,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            trust_remote_code=True,
            device_map="cuda:0",
        )
        # noinspection PyTypeChecker
        self._yasem = yasem.SpladeEmbedder(_SPARSE_MODEL, device="cuda:0")
        self._yasem.model = self.splade

    def close(self) -> None:
        self.qdrant_db.close()

    def _encode_dense(self, texts: list[str]) -> np.ndarray:
        encoded = self.ruri_tokenizer(
            texts, padding=True, truncation=True, max_length=8192, return_tensors="pt",
        )
        if hasattr(self.embedding, "device"):
            encoded = {k: v.to(self.embedding.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.embedding(**encoded)
        emb = _mean_pooling(output, encoded["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().float().numpy()

    def _encode_sparse(self, texts: list[str]) -> list[models.SparseVector]:
        sparse_matrix = self._yasem.encode(texts, convert_to_csr_matrix=True)
        return [
            models.SparseVector(
                indices=sparse_matrix.getrow(i).indices.tolist(),
                values=sparse_matrix.getrow(i).data.tolist(),
            )
            for i in range(sparse_matrix.shape[0])
        ]

    def search_medical_term(self, _term: str) -> str:
        dense_vec = self._encode_dense([f"検索クエリ: {_term}"])[0].tolist()
        sparse_vec = self._encode_sparse([_term])[0]

        results = self.qdrant_db.query_points_groups(
            collection_name=_COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=dense_vec, using="", limit=20),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices, values=sparse_vec.values,
                    ),
                    using="splade",
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),  # type: ignore[arg-type]
            group_by="article_id",
            group_size=3,
            limit=5,
            with_payload=True,
        )

        hits = []
        for group in results.groups:
            chunks = []
            title = None
            for point in group.hits:
                p = point.payload or {}
                if title is None:
                    title = p.get("title")
                chunks.append({
                    "chunk_text": p.get("chunk_text"),
                    "chunk_index": p.get("chunk_index"),
                    "score": point.score,
                })
            chunks.sort(key=lambda c: c.get("chunk_index", 0))
            hits.append({
                "article_id": group.id,
                "title": title,
                "chunks": chunks,
                "ja_title": None,
                "ja_url": None,
            })

        return json.dumps({"results": hits}, ensure_ascii=False, indent=2)


# ======================================================================
# エントリーポイント
# ======================================================================

if __name__ == "__main__":
    run_parallel_registration()

    searcher = MedicalTermSearcher()
    try:
        print(searcher.search_medical_term("juxtaglomerular apparatus"))
    finally:
        searcher.close()
