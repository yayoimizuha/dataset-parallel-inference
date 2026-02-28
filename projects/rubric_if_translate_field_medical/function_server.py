"""
Function Calling 用の FastAPI サーバー。

search_articles / get_ja_article を HTTP エンドポイントとして提供する。
Elasticsearch・DuckDB へのアクセスはこのプロセスに集約し、
task.py 側は純粋な HTTP リクエストだけで済むようにする。

起動例:
    uvicorn projects.rubric_if_translate_field_medical.function_server:app \
        --host 0.0.0.0 --port 8100

環境変数 (.env):
    ES_HOST  – Elasticsearch ホスト (default: localhost)
    ES_PORT  – Elasticsearch ポート (default: 9200)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import duckdb
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field

# ======================================================================
# .env 読み込み
# ======================================================================

load_dotenv(Path(__file__).parent / ".env")

# ======================================================================
# ロガー設定
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("function_server")

# ======================================================================
# 定数
# ======================================================================

_ES_HOST = os.environ.get("ES_HOST", "localhost")
_ES_PORT = int(os.environ.get("ES_PORT", "9200"))
_ES_INDEX_NAME = "wikipedia_medical"
_LANGLINKS_DB = Path(__file__).parent / "en_ja_langlinks.duckdb"
_TEXT_LIMIT = 3000

# ======================================================================
# バックエンド接続 (プロセス起動時に 1 回だけ)
# ======================================================================

_es_client = Elasticsearch(
    hosts=[{"host": _ES_HOST, "port": _ES_PORT, "scheme": "http"}],
    request_timeout=30,
)

_duckdb_con = duckdb.connect(str(_LANGLINKS_DB), read_only=True)

# ======================================================================
# リクエスト / レスポンス モデル
# ======================================================================


class SearchRequest(BaseModel):
    keyword: str
    size: int = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    article_id: int
    title: str
    text_snippet: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchHit]


class JaArticleRequest(BaseModel):
    article_id: int


class JaArticleResponse(BaseModel):
    article_id: int
    ja_title: str
    ja_text: str
    error: str | None = None


# ======================================================================
# FastAPI アプリケーション
# ======================================================================

app = FastAPI(title="Function Calling Server", version="1.0.0")


@app.middleware("http")
async def access_log(request: Request, call_next) -> Response:
    """全リクエストの処理時間・ステータスコードをログ出力する。"""
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s %d %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search_articles", response_model=SearchResponse)
def search_articles(req: SearchRequest):
    """Elasticsearch で英語版 Wikipedia 医学記事を全文検索する。"""
    logger.info("search_articles  keyword=%r  size=%d", req.keyword, req.size)
    query = {
        "query": {
            "multi_match": {
                "query": req.keyword,
                "fields": ["title^3", "text"],
                "type": "best_fields",
            },
        },
        "size": req.size,
        "_source": ["article_id", "title", "text"],
    }
    resp = _es_client.search(index=_ES_INDEX_NAME, body=query)
    hits = resp["hits"]["hits"]

    results = []
    for hit in hits:
        src = hit["_source"]
        text_full = src.get("text", "")
        snippet = text_full[:_TEXT_LIMIT] + ("..." if len(text_full) > _TEXT_LIMIT else "")
        results.append(SearchHit(
            article_id=int(src["article_id"]),
            title=src["title"],
            text_snippet=snippet,
            score=hit["_score"],
        ))
    logger.info("search_articles  keyword=%r  hits=%d", req.keyword, len(results))
    return SearchResponse(results=results)


@app.post("/get_ja_article", response_model=JaArticleResponse)
def get_ja_article(req: JaArticleRequest):
    """英語 Wikipedia 記事 ID から対応する日本語版 Wikipedia 記事を取得する。"""
    logger.info("get_ja_article  article_id=%d", req.article_id)
    cur = _duckdb_con.cursor()
    try:
        row = cur.execute(
            "SELECT ll_title FROM langlinks WHERE ll_from = ?",
            [req.article_id],
        ).fetchone()

        if row is None:
            logger.warning("get_ja_article  article_id=%d  langlink not found", req.article_id)
            return JaArticleResponse(
                article_id=req.article_id,
                ja_title="",
                ja_text="",
                error=f"article_id={req.article_id} に対応する日本語記事が見つかりません。",
            )
        ja_title = row[0]

        article_row = cur.execute(
            "SELECT id, title, text FROM ja_articles WHERE title = ?",
            [ja_title],
        ).fetchone()

        if article_row is None:
            logger.warning("get_ja_article  article_id=%d  ja_title=%r  article body not found", req.article_id, ja_title)
            return JaArticleResponse(
                article_id=req.article_id,
                ja_title=ja_title,
                ja_text="(日本語記事の本文は取得できませんでした)",
            )

        ja_text = article_row[2]
        ja_text = ja_text[:_TEXT_LIMIT] + ("..." if len(ja_text) > _TEXT_LIMIT else "")
        logger.info("get_ja_article  article_id=%d  ja_title=%r  text_len=%d", req.article_id, article_row[1], len(ja_text))
        return JaArticleResponse(
            article_id=req.article_id,
            ja_title=article_row[1],
            ja_text=ja_text,
        )
    finally:
        cur.close()


# ======================================================================
# 直接実行
# ======================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)
