import asyncio
import json
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

_BATCH_SIZE = 512
_CHUNK_SIZE = 512


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
            device="cuda:0",
        )

        self.splade = AutoModelForMaskedLM.from_pretrained(
            "hotchpotch/japanese-splade-v2",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            trust_remote_code=True,
            device="cuda:1",
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

    def _flush_batch(self, batch: list[dict]) -> None:
        """
        バッチを dense + sparse でエンべディングして Qdrant に upsert する。
        """
        chunk_texts = [item["chunk_text"] for item in batch]
        prefixed_texts = [f"検索文書: {t}" for t in chunk_texts]

        dense_vecs = self._encode_dense(prefixed_texts)
        sparse_vecs = self._encode_sparse(chunk_texts)

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
        self.qdrant_db.upsert("medical_terms", points=points)

    async def search_medical_term(self, _term: str) -> str:
        """
        英語の医療用語を指定すると、該当のWikipediaの記事と、それに対応する日本語Wikipediaの記事を返します。
        """
        query_prefixed = f"検索クエリ: {_term}"

        dense_vec = self._encode_dense([query_prefixed])[0].tolist()
        sparse_vec = self._encode_sparse([_term])[0]

        results = self.qdrant_db.query_points(
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
            limit=5,
            with_payload=True,
        )

        hits = []
        for point in results.points:
            p = point.payload or {}
            hits.append({
                "title": p.get("title"),
                "chunk_text": p.get("chunk_text"),
                "chunk_index": p.get("chunk_index"),
                "score": point.score,
                "ja_title": None,
                "ja_url": None,
            })

        return json.dumps({"results": hits}, ensure_ascii=False, indent=2)

    async def register_medical_terms(self) -> None:
        """
        医療用語をWikipediaから抽出し、Qdrantに登録します。
        """
        _len = self.wiki_dataset.info.splits["train"].num_examples  # type: ignore[union-attr]
        _batch: list[dict] = []

        _item: dict
        for _item in tqdm.tqdm(self.wiki_dataset, total=_len, desc="Registering"):  # type: ignore[assignment]
            _data_id: str = _item["id"]
            _title: str = _item["title"]
            _text: str = _item["text"]

            # 記事の全チャンクIDを生成して既登録チェック
            _chunks: list[str] = self.chunker(_text)
            _point_ids = [
                f"{_data_id}_{_chunk_idx}"
                for _chunk_idx in range(len(_chunks))
            ]
            _existing = self.qdrant_db.retrieve(
                "medical_terms",
                ids=_point_ids,
                with_payload=False,
                with_vectors=False,
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
                    self._flush_batch(_batch)
                    _batch.clear()

        if _batch:
            self._flush_batch(_batch)


if __name__ == '__main__':
    searcher = MedicalTermSearcher()
    asyncio.run(searcher.register_medical_terms())
    asyncio.run(searcher.search_medical_term("juxtaglomerular apparatus"))
