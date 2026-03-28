import json
import os
import sqlite3
import asyncio
from os import path
from os.path import dirname
import tqdm
from datasets import load_dataset
from dotenv import load_dotenv
from langfuse.openai import AsyncOpenAI
from openai import OpenAIError
from langfuse import get_client

from core import InferenceTask
from asyncio import Semaphore

MAX_AGENT_TURNS = 5
MAX_VERIFY_RETRIES = 3

SUBAGENT_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "call_subagent",
        "description": (
            "別のAIエージェントに特定のタスク（翻訳結果のチェック、特定箇所の詳細な考察、"
            "要約、表現の改善提案など）を依頼する。サブエージェントは独立したリクエストとして実行される。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "サブエージェントへの指示内容。具体的なタスクを記述する。",
                },
                "include_full_context": {
                    "type": "boolean",
                    "description": (
                        "trueの場合、これまでの全翻訳済みフィールド（原文・訳文のペア）をサブエージェントに渡す。"
                        "falseの場合、target_contentのみを渡す。"
                    ),
                },
                "target_content": {
                    "type": "string",
                    "description": (
                        "サブエージェントが処理すべき対象テキスト。"
                        "include_full_contextがfalseの場合に指定する。"
                    ),
                },
            },
            "required": ["task", "include_full_context"],
        },
    },
}

VERIFY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "verify_translation",
        "schema": {
            "type": "object",
            "properties": {
                "ok": {
                    "type": "boolean",
                    "description": "翻訳が妥当であればtrue、問題があればfalse。",
                },
                "reason": {
                    "type": "string",
                    "description": "判定理由。okがtrueの場合も簡潔に根拠を述べること。",
                },
            },
            "required": ["ok", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


class Task(InferenceTask):
    def __init__(self):
        self._db = sqlite3.connect(path.join(dirname(__file__), "db.sqlite"))
        self._cur = self._db.cursor()
        self.dataset = load_dataset("nohurry/Opus-4.6-Reasoning-3000x-filtered", streaming=False)["train"]
        self._cur.execute("""
            CREATE TABLE IF NOT EXISTS result(
                id       INT  PRIMARY KEY,
                content  TEXT,
                reason   TEXT,
                verify   TEXT,
                subagent TEXT,
                source   TEXT
            );
        """)
        load_dotenv(path.join(dirname(__file__), ".env"))
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"], timeout=None)
        self._db_lock = asyncio.Lock()

    def get_length(self) -> int:
        return self.dataset.info.splits["train"].num_examples

    def __del__(self):
        self._db.commit()
        self._cur.close()
        self._db.close()

    async def close(self) -> None:
        get_client().flush()

    async def _call_subagent(
        self,
        task: str,
        include_full_context: bool,
        target_content: str | None,
        original_fields: dict[str, str],
        content_fields: dict[str, str],
    ) -> dict:
        """サブエージェントを呼び出し、{"task", "include_full_context", "target_content", "result", "reasoning_content"} を返す。"""
        if include_full_context:
            context_block = "\n\n\n".join(
                f"===={field}=============\n"
                f"{original_fields[field]}"
                f"\n\n-------↓↓↓↓↓↓-------\n\n"
                f"{content_fields.get(field, '(未翻訳)')}"
                "\n============================="
                for field in ("problem", "thinking", "solution")
                if original_fields.get(field)
            )
            subagent_input = (
                "以下は原文と翻訳のペア一覧です:\n\n\n" +
                context_block +
                "\n\n\n\n" +
                task
            )
        else:
            content = target_content or ""
            subagent_input = (
                f"以下のテキストに対してタスクを実行してください:\n\n{content}\n\n\n{task}"
            )

        resp = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": subagent_input}],
            model=os.environ["MODEL_NAME"],
            extra_body={"separate_reasoning": True},
            reasoning_effort="high",
        )
        msg = resp.choices[0].message
        return {
            "task": task,
            "include_full_context": include_full_context,
            "target_content": target_content,
            "result": msg.content or "",
            "reasoning_content": getattr(msg, "reasoning_content", None) or "",
        }

    async def _verify_translation(
        self,
        field: str,
        original: str,
        translated: str,
        feedback: str | None,
    ) -> dict:
        """翻訳を検証し、{"ok": bool, "reason": str, "reasoning_content": str} を返す。"""
        prompt = (
            f"以下の原文（フィールド: {field}）と翻訳文のペアを検証してください。\n\n"
            "【判定基準】\n"
            " - 原文の情報が欠落・付加なく翻訳されているか\n"
            " - 原文の雰囲気・文脈が適切に反映されているか\n"
            " - 自然な日本語になっているか\n"
        )
        if feedback:
            prompt += f"\n【前回の指摘事項】\n{feedback}\n"
        prompt += (
            f"\n===原文===\n{original}\n\n"
            f"===翻訳===\n{translated}\n"
        )

        resp = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=os.environ["MODEL_NAME"],
            extra_body={"separate_reasoning": True},
            reasoning_effort="high",
            response_format=VERIFY_RESPONSE_FORMAT,
        )
        msg = resp.choices[0].message
        parsed = json.loads(msg.content or "{}")
        return {
            "ok": parsed.get("ok", False),
            "reason": parsed.get("reason", ""),
            "reasoning_content": getattr(msg, "reasoning_content", None) or "",
        }

    async def _translate_field(
        self,
        field: str,
        text: str,
        context_block: str,
        original_fields: dict[str, str],
        content_fields: dict[str, str],
        subagent_fields: dict[str, list[dict]],
        feedback: str | None = None,
    ) -> tuple[str, str]:
        """1フィールドを翻訳し (content, reasoning_content) を返す。"""
        chat_string = (
            ("過去の翻訳済みフィールド(一貫性のある翻訳のためのコンテキスト):\n\n\n"
             + context_block + "\n\n\n\n")
            if context_block else ""
        ) + (
            (f"【前回の検証での指摘事項】\n{feedback}\n上記の指摘を踏まえて再翻訳してください。\n\n")
            if feedback else ""
        ) + (
            f"以下に外国語の文章A（フィールド: {field}）が与えられます。"
            "その文章を**全て**日本語に翻訳してください。なお、以下の条件を**遵守**すること。\n"
            "\n"
            " - 原文に忠実に翻訳し、原文に存在する情報を欠落させたり、書かれていないことを付け加えないこと。\n"
            " - 原文の雰囲気や文脈に基づいて翻訳すること。\n"
            " - 翻訳済みの文章のみを出力し、余計な説明や注釈を加えないこと。\n"
            " - 難しい箇所があった時の推敲、翻訳の点検としてサブエージェントを活用すること。\n\n"
            "\n===文章A==========================\n\n\n"
            + text
        )

        messages = [{"role": "user", "content": chat_string}]
        final_content = None
        final_reasoning = None

        for _turn in range(MAX_AGENT_TURNS):
            resp = await self._client.chat.completions.create(
                messages=messages,
                model=os.environ["MODEL_NAME"],
                extra_body={"separate_reasoning": True},
                reasoning_effort="high",
                tools=[SUBAGENT_TOOL_DEF],
                tool_choice="auto",
            )
            assistant_message = resp.choices[0].message
            messages.append(assistant_message.to_dict())

            tool_calls = assistant_message.tool_calls
            if not tool_calls:
                final_content   = assistant_message.content or ""
                final_reasoning = getattr(assistant_message, "reasoning_content", None) or ""
                break

            for tc in tool_calls:
                args = json.loads(tc.function.arguments)
                subagent_record = await self._call_subagent(
                    task=args.get("task", ""),
                    include_full_context=args.get("include_full_context", False),
                    target_content=args.get("target_content"),
                    original_fields=original_fields,
                    content_fields=content_fields,
                )
                subagent_fields[field].append(subagent_record)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": subagent_record["result"],
                })

        if final_content is None:
            # MAX_AGENT_TURNS 到達 → ツール定義を渡さず最終回答を得る
            resp = await self._client.chat.completions.create(
                messages=messages,
                model=os.environ["MODEL_NAME"],
                extra_body={"separate_reasoning": True},
                reasoning_effort="high",
            )
            msg = resp.choices[0].message
            final_content   = msg.content or "<-- output is missing -->"
            final_reasoning = getattr(msg, "reasoning_content", None) or ""

        return final_content, final_reasoning

    async def process(self, data, order: int, sem: Semaphore, bar: tqdm.tqdm):
        # id列に order の値が存在するか確認、したらスキップ
        async with self._db_lock:
            if self._cur.execute("SELECT COUNT(*) FROM result WHERE id=?;", (order,)).fetchone()[0] > 0:
                bar.update(1)
                return
        async with sem:
            original_fields = {
                "problem":  data["problem"]  or "",
                "thinking": data["thinking"] or "",
                "solution": data["solution"] or "",
            }
            content_fields:  dict[str, str]        = {}
            reason_fields:   dict[str, str]        = {}
            verify_fields:   dict[str, list[dict]] = {
                "problem": [], "thinking": [], "solution": []
            }
            subagent_fields: dict[str, list[dict]] = {
                "problem": [], "thinking": [], "solution": []
            }

            for field in ("problem", "thinking", "solution"):
                text = original_fields[field]
                if not text:
                    content_fields[field] = "<---input is empty--->"
                    reason_fields[field]  = ""
                    continue

                context_block = "\n\n\n".join(
                    f"===={f}=============\n"
                    f"{original_fields[f]}"
                    f"\n\n-------↓↓↓↓↓↓-------\n\n"
                    f"{content_fields[f]}"
                    "\n============================="
                    for f in ("problem", "thinking", "solution")
                    if f in content_fields and content_fields[f]
                )

                sleep_time = 4.0
                while True:
                    try:
                        feedback: str | None = None

                        for attempt in range(MAX_VERIFY_RETRIES):
                            # 翻訳
                            final_content, final_reasoning = await self._translate_field(
                                field=field,
                                text=text,
                                context_block=context_block,
                                original_fields=original_fields,
                                content_fields=content_fields,
                                subagent_fields=subagent_fields,
                                feedback=feedback,
                            )

                            # 検証
                            verify_result = await self._verify_translation(
                                field=field,
                                original=text,
                                translated=final_content,
                                feedback=feedback,
                            )
                            verify_result["attempt"] = attempt
                            verify_fields[field].append(verify_result)

                            if verify_result["ok"]:
                                break

                            # NGなら指摘内容をフィードバックして再試行
                            feedback = verify_result["reason"]

                        content_fields[field] = final_content
                        reason_fields[field]  = final_reasoning
                        break
                    except OpenAIError as e:
                        if sleep_time > 16.0:
                            content_fields[field] = "<-- output is missing -->"
                            reason_fields[field]  = ""
                            break
                        print(f"OpenAI API Error: {e}")
                        await asyncio.sleep(sleep_time)
                        sleep_time *= 2

        async with self._db_lock:
            self._cur.execute(
                "REPLACE INTO result(id, content, reason, verify, subagent, source) VALUES (?,?,?,?,?,?);",
                (
                    order,
                    json.dumps(content_fields,  ensure_ascii=False),
                    json.dumps(reason_fields,   ensure_ascii=False),
                    json.dumps(verify_fields,   ensure_ascii=False),
                    json.dumps(subagent_fields, ensure_ascii=False),
                    json.dumps(original_fields, ensure_ascii=False),
                ),
            )
            self._db.commit()
        bar.update(1)
