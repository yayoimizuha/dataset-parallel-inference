"""
ElasticSearch full-text search index builder for Wikipedia articles.

Pipeline
========
1. Load Wikipedia dataset via HuggingFace ``datasets``
2. Preprocess articles with multiprocessing.Pool
3. Bulk-index into ElasticSearch using ``elasticsearch.helpers.bulk``

Environment variables (in .env):
    ES_HOST  – ElasticSearch host  (default: localhost)
    ES_PORT  – ElasticSearch port  (default: 9200)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import uuid
from pathlib import Path

import duckdb
import tqdm
from datasets import load_dataset
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# ======================================================================
# .env 読み込み
# ======================================================================

load_dotenv(Path(__file__).parent / ".env")

# ======================================================================
# 定数
# ======================================================================

_UUID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

_BULK_BATCH_SIZE = 500  # ES バルクインサートのバッチサイズ
_NUM_PROC = int(max(4, (os.cpu_count() or 1) / 2 - 10))  # 並列ワーカー数

_DATASET_NAME = "omarkamali/wikipedia-monthly"
_DATASET_CONFIG = "latest.en"

_ES_HOST = os.environ.get("ES_HOST", "localhost")
_ES_PORT = int(os.environ.get("ES_PORT", "9200"))
_ES_INDEX_NAME = "wikipedia_medical"

# ======================================================================
# インデックスマッピング定義
# ======================================================================

_INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 3,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "english_analyzer": {
                    "type": "standard",
                    "stopwords": "_english_",
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "article_id": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "english_analyzer",
                "fields": {
                    "keyword": {"type": "keyword"},
                },
            },
            "text": {
                "type": "text",
                "analyzer": "english_analyzer",
            },
        },
    },
}

# ======================================================================
# 前処理 (multiprocessing ワーカー)
# ======================================================================


def _preprocess_article(article: dict) -> dict:
    """
    1 記事を ES ドキュメントに変換する。multiprocessing.Pool のワーカー関数。
    """
    data_id = article["id"]
    title = article["title"]
    text = article["text"]

    doc_id = str(uuid.uuid5(_UUID_NAMESPACE, str(data_id)))

    return {
        "_index": _ES_INDEX_NAME,
        "_id": doc_id,
        "_source": {
            "article_id": data_id,
            "title": title,
            "text": text,
        },
    }


# ======================================================================
# バルクインデックス
# ======================================================================


def _bulk_index(
    es: Elasticsearch,
    actions: list[dict],
) -> tuple[int, int]:
    """
    actions を ES にバルクインデックスし、(success, failed) を返す。
    """
    success, errors = bulk(
        es,
        actions,
        raise_on_error=False,
        raise_on_exception=False,
    )
    failed = len(errors) if isinstance(errors, list) else 0
    return success, failed


# ======================================================================
# メインパイプライン
# ======================================================================


def main() -> None:
    mp.set_start_method("spawn", force=True)

    # ------------------------------------------------------------------
    # ES 接続
    # ------------------------------------------------------------------
    es = Elasticsearch(
        hosts=[{"host": _ES_HOST, "port": _ES_PORT, "scheme": "http"}],
        request_timeout=60,
    )
    info = es.info()
    print(f"[Setup] Connected to ElasticSearch: {info['version']['number']}", flush=True)

    # ------------------------------------------------------------------
    # インデックス作成 (存在しなければ)
    # ------------------------------------------------------------------
    if es.indices.exists(index=_ES_INDEX_NAME):
        print(f"[Setup] Index '{_ES_INDEX_NAME}' already exists. Deleting...", flush=True)
        es.indices.delete(index=_ES_INDEX_NAME)

    es.indices.create(index=_ES_INDEX_NAME, body=_INDEX_SETTINGS)
    print(f"[Setup] Index '{_ES_INDEX_NAME}' created.", flush=True)

    # ------------------------------------------------------------------
    # データセットロード
    # ------------------------------------------------------------------
    print(f"[Pipeline] Loading dataset: {_DATASET_NAME} ({_DATASET_CONFIG})...", flush=True)
    wiki = load_dataset(_DATASET_NAME, _DATASET_CONFIG, split="train", streaming=False)
    total_articles = len(wiki)
    print(f"[Pipeline] Total articles: {total_articles}", flush=True)

    # ------------------------------------------------------------------
    # multiprocessing.Pool で前処理 + バルクインデックス
    # ------------------------------------------------------------------
    total_success = 0
    total_failed = 0
    batch_buf: list[dict] = []

    progress = tqdm.tqdm(total=total_articles, desc="[Pipeline] Indexing")

    with mp.Pool(processes=_NUM_PROC) as pool:
        for doc in pool.imap_unordered(_preprocess_article, wiki, chunksize=256):
            batch_buf.append(doc)
            progress.update(1)

            if len(batch_buf) >= _BULK_BATCH_SIZE:
                success, failed = _bulk_index(es, batch_buf)
                total_success += success
                total_failed += failed
                batch_buf = []

    # 残余フラッシュ
    if batch_buf:
        success, failed = _bulk_index(es, batch_buf)
        total_success += success
        total_failed += failed

    progress.close()

    # ------------------------------------------------------------------
    # 結果表示
    # ------------------------------------------------------------------
    es.indices.refresh(index=_ES_INDEX_NAME)
    count = es.count(index=_ES_INDEX_NAME)["count"]
    print(
        f"[Pipeline] Done. Indexed: {total_success}, Failed: {total_failed}, "
        f"Total docs in index: {count}",
        flush=True,
    )


# ======================================================================
# DuckDB: 英日記事タイトル対応
# ======================================================================

_LANGLINKS_DB = Path(__file__).parent / "en_ja_langlinks.duckdb"


def _lookup_ja_titles(article_ids: list[str]) -> dict[str, str]:
    """
    英語 Wikipedia 記事ID のリストを受け取り、
    DuckDB の langlinks テーブルから対応する日本語タイトルを引く。

    Returns:
        {article_id: ja_title} の辞書。対応がない ID は含まれない。
    """
    if not article_ids:
        return {}

    con = duckdb.connect(str(_LANGLINKS_DB), read_only=True)
    try:
        # パラメータバインドのため一時テーブルに挿入
        int_ids = [int(aid) for aid in article_ids]
        result = con.execute(
            "SELECT ll_from, ll_title FROM langlinks WHERE ll_from IN "
            f"({','.join('?' for _ in int_ids)})",
            int_ids,
        ).fetchall()
        return {str(row[0]): row[1] for row in result}
    finally:
        con.close()


# ======================================================================
# 検索テスト関数
# ======================================================================


def _get_es_client() -> Elasticsearch:
    """共通の ES クライアントを返す。"""
    return Elasticsearch(
        hosts=[{"host": _ES_HOST, "port": _ES_PORT, "scheme": "http"}],
        request_timeout=30,
    )


def test_search(keyword: str, size: int = 10) -> None:
    """
    ES 全文検索テスト。

    1. keyword で wikipedia_medical インデックスを検索
    2. ヒットした英語記事に対応する日本語タイトルを DuckDB から取得
    3. 結果を表示

    Args:
        keyword: 検索キーワード (例: "juxtaglomerular apparatus")
        size: 最大取得件数
    """
    es = _get_es_client()

    # multi_match: title と text の両方を検索
    query = {
        "query": {
            "multi_match": {
                "query": keyword,
                "fields": ["title^3", "text"],
                "type": "best_fields",
            },
        },
        "size": size,
        "_source": ["article_id", "title"],
    }

    resp = es.search(index=_ES_INDEX_NAME, body=query)
    hits = resp["hits"]["hits"]

    if not hits:
        print(f"No results for: {keyword}")
        return

    # 日本語タイトル一括ルックアップ
    article_ids = [hit["_source"]["article_id"] for hit in hits]
    ja_titles = _lookup_ja_titles(article_ids)

    print(f"Search results for: \"{keyword}\" ({len(hits)} hits)\n")
    print(f"{'#':>3}  {'Score':>7}  {'Article ID':>12}  {'EN Title':<40}  {'JA Title'}")
    print("-" * 120)

    for i, hit in enumerate(hits, 1):
        score = hit["_score"]
        src = hit["_source"]
        aid = src["article_id"]
        en_title = src["title"]
        ja_title = ja_titles.get(str(aid), "(対応なし)")
        print(f"{i:>3}  {score:>7.2f}  {aid:>12}  {en_title:<40}  {ja_title}")

    print()


# ======================================================================
# DuckDB: 日本語 Wikipedia 記事取り込み
# ======================================================================

_JA_DATASET_CONFIG = "latest.ja"


def load_ja_articles() -> None:
    """
    ``omarkamali/wikipedia-monthly`` の ``latest.ja`` をダウンロードし、
    DuckDB の ``ja_articles`` テーブルに追記する。

    テーブルスキーマ:
        id      VARCHAR  -- Wikipedia 記事 ID
        url     VARCHAR  -- 記事 URL
        title   VARCHAR  -- 記事タイトル
        text    VARCHAR  -- 記事本文

    既にテーブルが存在する場合はドロップしてから再作成する。
    Dataset → Arrow 変換を利用し、DuckDB に直接取り込むことで高速化。
    """
    print(f"[JA] Loading dataset: {_DATASET_NAME} ({_JA_DATASET_CONFIG})...", flush=True)
    ja_wiki = load_dataset(
        _DATASET_NAME, _JA_DATASET_CONFIG, split="train", streaming=False,
    )
    total_articles = len(ja_wiki)
    print(f"[JA] Total articles: {total_articles}", flush=True)

    con = duckdb.connect(str(_LANGLINKS_DB))
    try:
        # テーブルを再作成
        con.execute("DROP TABLE IF EXISTS ja_articles")

        # HuggingFace Dataset の内部 Arrow テーブルを直接取り込み
        print("[JA] Converting to Arrow and inserting into DuckDB...", flush=True)
        arrow_table = ja_wiki.data.table  # type: ignore[union-attr]
        con.execute("CREATE TABLE ja_articles AS SELECT * FROM arrow_table")

        # 確認
        row_count = con.execute("SELECT COUNT(*) FROM ja_articles").fetchone()[0]
        print(f"[JA] Done. Total rows in 'ja_articles': {row_count}", flush=True)

        # title にインデックス作成 (langlinks.ll_title との結合高速化)
        con.execute("CREATE INDEX IF NOT EXISTS idx_ja_title ON ja_articles (title)")
        print("[JA] Index on 'title' created.", flush=True)

    finally:
        con.close()


# ======================================================================
# エントリーポイント
# ======================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_search(" ".join(sys.argv[1:]))
    else:
        # main()  # ES 登録済みのためスキップ
        load_ja_articles()