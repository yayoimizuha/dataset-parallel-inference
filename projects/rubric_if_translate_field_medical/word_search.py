import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import semchunk
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
import tqdm
import yasem
from datasets import load_dataset
from qdrant_client import QdrantClient, models
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer, BitsAndBytesConfig

_BATCH_SIZE = 4096
_CHUNK_SIZE = 512
_UUID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

# パイプラインキューの最大サイズ（バッチ数）
# メモリ使用量を制限しつつ、ステージ間の待ちを減らす
_QUEUE_MAXSIZE = 3


def _mean_pooling(model_output, attention_mask) -> torch.Tensor:
    token_embeddings = model_output.last_hidden_state  # [B, seq_len, 768]
    mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
    sum_embeddings = (token_embeddings * mask_expanded).sum(1)  # [B, 768]
    sum_mask = mask_expanded.sum(1).clamp(min=1e-9)  # [B, 1]
    return sum_embeddings / sum_mask  # [B, 768]


class MedicalTermSearcher:
    def __init__(self):
        self.qdrant_db = QdrantClient(path=Path(__file__).parent.joinpath("qdrant_db").as_posix())
        if not self.qdrant_db.collection_exists("medical_terms"):
            self.qdrant_db.create_collection(
                "medical_terms",
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE
                ),
                sparse_vectors_config={"splade": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        full_scan_threshold=1000,
                    )
                )}
            )
        self.wiki_dataset = load_dataset("omarkamali/wikipedia-monthly", "latest.en", split="train", streaming=False)

        # noinspection PyNoneFunctionAssignment
        self.ruri_tokenizer = AutoTokenizer.from_pretrained("cl-nagoya/ruri-v3-310m")

        self.embedding = AutoModel.from_pretrained(
            "cl-nagoya/ruri-v3-310m",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            trust_remote_code=True,
            device_map="cuda:0",
        )

        self.splade = AutoModelForMaskedLM.from_pretrained(
            "hotchpotch/japanese-splade-v2",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            trust_remote_code=True,
            device_map="cuda:1",
        )
        # noinspection PyTypeChecker
        self._yasem = yasem.SpladeEmbedder(
            "hotchpotch/japanese-splade-v2",
            device="cuda:1"
        )
        self._yasem.model = self.splade

        # noinspection PyNoneFunctionAssignment
        self.splade_tokenizer = AutoTokenizer.from_pretrained("hotchpotch/japanese-splade-v2")
        # noinspection PyTypeChecker
        self.chunker = semchunk.chunkerify(self.splade_tokenizer, chunk_size=_CHUNK_SIZE)

        # GPU 推論は同一スレッドで逐次実行（torch.no_grad のスレッド安全性のため）
        self._gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self._io_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="io")

    def close(self) -> None:
        """リソースを明示的に解放する。"""
        self._gpu_executor.shutdown(wait=False)
        self._io_executor.shutdown(wait=False)
        self.qdrant_db.close()

    # ------------------------------------------------------------------
    # エンコード（同期メソッド — GPU スレッドから呼ばれる）
    # ------------------------------------------------------------------

    def _encode_dense(self, texts: list[str]) -> np.ndarray:
        """
        ruri-v3-310m でテキストをdenseエンべディングする。
        戻り値: shape (N, 768) の numpy 配列（L2正規化済み）
        """
        encoded = self.ruri_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        if hasattr(self.embedding, "device"):
            encoded = {k: v.to(self.embedding.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.embedding(**encoded)
        emb = _mean_pooling(output, encoded["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().float().numpy()

    def _encode_sparse(self, texts: list[str]) -> list[models.SparseVector]:
        """
        japanese-splade-v2 でテキストをsparseエンべディングする。
        戻り値: Qdrant SparseVector のリスト
        """
        sparse_matrix = self._yasem.encode(texts, convert_to_csr_matrix=True)
        result = []
        for i in range(sparse_matrix.shape[0]):
            row = sparse_matrix.getrow(i)
            result.append(models.SparseVector(
                indices=row.indices.tolist(),
                values=row.data.tolist(),
            ))
        return result

    def _encode_both(
        self, dense_texts: list[str], sparse_texts: list[str]
    ) -> tuple[np.ndarray, list[models.SparseVector]]:
        """
        Dense + Sparse を同一スレッドで逐次実行する。
        torch.no_grad() のグローバル状態が競合しないことを保証する。
        """
        dense_vecs = self._encode_dense(dense_texts)
        sparse_vecs = self._encode_sparse(sparse_texts)
        return dense_vecs, sparse_vecs

    # ------------------------------------------------------------------
    # 非同期ラッパー
    # ------------------------------------------------------------------

    async def _encode_both_async(
        self, dense_texts: list[str], sparse_texts: list[str]
    ) -> tuple[np.ndarray, list[models.SparseVector]]:
        """Dense + Sparse 推論を GPU スレッドで非同期実行する。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._gpu_executor, self._encode_both, dense_texts, sparse_texts
        )

    # ------------------------------------------------------------------
    # Qdrant upsert（非同期）
    # ------------------------------------------------------------------

    def _upsert_points(self, points: list[models.PointStruct]) -> None:
        """Qdrant への upsert（同期 — IOスレッドから呼ばれる）。"""
        self.qdrant_db.upsert("medical_terms", points=points)

    async def _upsert_points_async(self, points: list[models.PointStruct]) -> None:
        """upsert を IO スレッドプールで非同期実行する。"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._io_executor, self._upsert_points, points)

    # ------------------------------------------------------------------
    # ポイント構築（CPU処理）
    # ------------------------------------------------------------------

    @staticmethod
    def _build_points(
        batch: list[dict],
        dense_vecs: np.ndarray,
        sparse_vecs: list[models.SparseVector],
    ) -> list[models.PointStruct]:
        """バッチ、Dense/Sparseベクトルから Qdrant PointStruct のリストを構築する。"""
        points = []
        for item, dense, sparse in zip(batch, dense_vecs, sparse_vecs):
            points.append(models.PointStruct(
                id=item["point_id"],
                vector={
                    "": dense.tolist(),
                    "splade": sparse,
                },
                payload={
                    "article_id": item["article_id"],
                    "title": item["title"],
                    "chunk_text": item["chunk_text"],
                    "chunk_index": item["chunk_index"],
                },
            ))
        return points

    # ------------------------------------------------------------------
    # 検索
    # ------------------------------------------------------------------

    async def search_medical_term(self, _term: str) -> str:
        """
        英語の医療用語を指定すると、該当のWikipediaの記事と、それに対応する日本語Wikipediaの記事を返します。
        """
        query_prefixed = f"検索クエリ: {_term}"

        dense_result, sparse_result = await self._encode_both_async(
            [query_prefixed], [_term]
        )

        dense_vec = dense_result[0].tolist()
        sparse_vec = sparse_result[0]

        results = self.qdrant_db.query_points_groups(
            collection_name="medical_terms",
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="",
                    limit=20,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
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

    # ------------------------------------------------------------------
    # 登録パイプライン
    # ------------------------------------------------------------------

    async def register_medical_terms(self) -> None:
        """
        医療用語をWikipediaから抽出し、Qdrantに登録する。

        3段パイプライン:
          Stage 1 (CPU)     : チャンク分割 → バッチ生成 → embed_queue へ
          Stage 2 (GPU)     : Dense + Sparse 推論（同一スレッドで逐次） → upsert_queue へ
          Stage 3 (IO)      : Qdrant upsert

        Stage 2 が GPU で推論している間に Stage 1 は次のバッチの前処理を進め、
        Stage 3 は前のバッチの upsert を IO スレッドで実行する。
        """
        embed_queue: asyncio.Queue[list[dict] | None] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        upsert_queue: asyncio.Queue[list[models.PointStruct] | None] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)

        # ----------------------------------------------------------
        # Stage 1: 前処理 (CPU) — チャンク分割・バッチ生成
        # ----------------------------------------------------------
        async def _stage_preprocess() -> None:
            _len = self.wiki_dataset.info.splits["train"].num_examples  # type: ignore[union-attr]
            _batch: list[dict] = []

            loop = asyncio.get_running_loop()

            _item: dict
            for _item in tqdm.tqdm(self.wiki_dataset, total=_len, desc="Registering"):  # type: ignore[assignment]
                _data_id: str = _item["id"]
                _title: str = _item["title"]
                _text: str = _item["text"]

                _chunks: list[str] = self.chunker(_text)
                _point_ids = [
                    str(uuid.uuid5(_UUID_NAMESPACE, f"{_data_id}_{_chunk_idx}"))
                    for _chunk_idx in range(len(_chunks))
                ]

                # 既登録チェックを IO スレッドへオフロード
                _existing = await loop.run_in_executor(
                    self._io_executor,
                    lambda ids=_point_ids: self.qdrant_db.retrieve(
                        "medical_terms",
                        ids=ids,
                        with_payload=False,
                        with_vectors=False,
                    ),
                )
                _existing_ids = {p.id for p in _existing}

                for _chunk_idx, (_chunk, _point_id) in enumerate(zip(_chunks, _point_ids)):
                    if _point_id in _existing_ids:
                        continue
                    _batch.append({
                        "point_id": _point_id,
                        "article_id": _data_id,
                        "title": _title,
                        "chunk_text": _chunk,
                        "chunk_index": _chunk_idx,
                    })
                    if len(_batch) >= _BATCH_SIZE:
                        await embed_queue.put(_batch)
                        _batch = []

            if _batch:
                await embed_queue.put(_batch)

            # 終了シグナル
            await embed_queue.put(None)

        # ----------------------------------------------------------
        # Stage 2: 推論 (GPU) — Dense + Sparse を同一スレッドで逐次実行
        # ----------------------------------------------------------
        async def _stage_embed() -> None:
            while True:
                batch = await embed_queue.get()
                if batch is None:
                    await upsert_queue.put(None)
                    break

                chunk_texts = [item["chunk_text"] for item in batch]
                prefixed_texts = [f"検索文書: {t}" for t in chunk_texts]

                dense_vecs, sparse_vecs = await self._encode_both_async(
                    prefixed_texts, chunk_texts
                )

                points = self._build_points(batch, dense_vecs, sparse_vecs)
                await upsert_queue.put(points)

        # ----------------------------------------------------------
        # Stage 3: Qdrant upsert (IO)
        # ----------------------------------------------------------
        async def _stage_upsert() -> None:
            while True:
                points = await upsert_queue.get()
                if points is None:
                    break
                await self._upsert_points_async(points)

        # パイプライン起動 — 全ステージを並行実行
        await asyncio.gather(
            _stage_preprocess(),
            _stage_embed(),
            _stage_upsert(),
        )


if __name__ == '__main__':
    searcher = MedicalTermSearcher()
    try:
        asyncio.run(searcher.register_medical_terms())
        asyncio.run(searcher.search_medical_term("juxtaglomerular apparatus"))
    finally:
        searcher.close()
