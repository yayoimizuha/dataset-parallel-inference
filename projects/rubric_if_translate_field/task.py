import glob
import json
import os
import sqlite3
import asyncio
from os import path
from os.path import dirname, basename
from pathlib import Path

import jsonpath_ng
import tqdm
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam
from pydantic import BaseModel
from core import InferenceTask
from asyncio import Semaphore


class TranslationElaboration(BaseModel):
    """Model for the translation elaboration/reasoning response."""
    elaboration: str


class TranslationResult(BaseModel):
    """Model for the final translation result."""
    translation: str


class Task(InferenceTask):
    def __init__(self):
        self._db = sqlite3.connect(Path(__file__).parent.parent.joinpath("rubric_if_define_field", "db.sqlite"))
        self._cur = self._db.cursor()
        self._cur.execute("CREATE TABLE IF NOT EXISTS translate(id INT PRIMARY KEY,content TEXT,reason TEXT);")
        load_dotenv(path.join(dirname(__file__), ".env"))
        self._client = AsyncOpenAI(api_key=os.environ["API_KEY"], base_url=os.environ["BASE_URL"], timeout=None)
        self.function_definitions = {basename(_file).removesuffix(".py") for _file in
                                     glob.glob(Path(__file__).parent.parent.joinpath(
                                         "rubric_if_define_field", "functions", "*.py"
                                     ).__str__())}
        self.dataset = range(self.get_length())
        load_dotenv(path.join(dirname(__file__), ".env"))

    def get_length(self) -> int:
        return self._cur.execute("SELECT COUNT(*) FROM result;").fetchone()[0]

    def __del__(self):
        self._db.commit()
        self._cur.close()
        self._db.close()

    async def process(self, data, order: int, sem: Semaphore, bar: tqdm.tqdm):
        # id列に order の値が存在するか確認、したらスキップ
        if self._cur.execute("SELECT COUNT(*) FROM translate WHERE id=?;", (order,)).fetchone()[0] > 0:
            bar.update(1)
            return
        async with sem:
            elaborate_prompt = """日本語への翻訳タスクです。以降、外国語の翻訳対象の文章が与えられます。その文章を日本語に翻訳するにあたって、注意すべきと思われる点を全ての条件について具体的にどのような訳語を用いるべきかについてまで、確実と言い切れるまで** 何度も **推敲してください。なお、議論にあたっては以下の条件を**遵守**すること。

 - 人名・固有名詞については翻訳せず、原文での表記のまま書くこと。
 - 原文に忠実に翻訳し、原文に存在する情報を欠落させたり、書かれていないことを付け加えないこと。
 - 原文の雰囲気や文脈に基づいて翻訳すること。
 - 推敲とは原文の文脈を分析し、次に多義語の選択肢を挙げ、最後に最も適切な表現を決定するプロセスを順を追って説明することです。
 - 最終的な翻訳結果自体は出力しないでください。"""
            input_string = json.loads(
                self._cur.execute("SELECT source FROM result WHERE id = ?;", (order,)).fetchone()[0]
            )
            if json.dumps(input_string, ensure_ascii=False).__len__() > 30000:
                bar.update(1)
                return
            _contents = []
            _reasons = []
            _positions = []
            for _translate_pos in json.loads(
                    self._cur.execute("SELECT content FROM result WHERE id = ?;", (order,)).fetchone()[0]
            ):
                _positions.append(_translate_pos)
                subject_txt = jsonpath_ng.parse(_translate_pos).find(input_string)[0]
                prompt = f"""以下のデータセットのうち、{_translate_pos}に該当する部分について処理します。

{json.dumps(input_string, ensure_ascii=False, indent=2)}

=======(翻訳の一貫性のための)翻訳履歴=========
{"\n".join(["\n=======" + _pos + "=========\n" + _cont for _cont, _pos in zip(_contents, _positions)])}
==============================
{elaborate_prompt}
======={_translate_pos}=======
{subject_txt}"""
                sleep_time = 4.0
                while True:
                    try:
                        resp_1 = await self._client.chat.completions.parse(
                            messages=[ChatCompletionUserMessageParam(
                                content=prompt,
                                role="user"
                            )],
                            model=os.environ["MODEL_NAME"],
                            extra_body={"separate_reasoning": True},
                            reasoning_effort="none",
                            response_format=TranslationElaboration,
                        )
                        if resp_1.choices[0].message.parsed is None:
                            raise OpenAIError("Failed to parse TranslationElaboration response from resp_1. Expected structured output with 'elaboration' field.")
                        
                        resp_2 = await self._client.chat.completions.parse(
                            messages=[
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": resp_1.choices[0].message.parsed.elaboration},
                                {"role": "user", "content": "推敲をもとに、全文の和訳のみを出力してください。"}
                            ],
                            model=os.environ["MODEL_NAME"],
                            extra_body={"separate_reasoning": True},
                            reasoning_effort="none",
                            response_format=TranslationResult,
                        )
                        if resp_2.choices[0].message.parsed is None:
                            raise OpenAIError("Failed to parse TranslationResult response from resp_2. Expected structured output with 'translation' field.")
                        break
                    except (OpenAIError, ValueError) as e:
                        if sleep_time > 32.0:
                            print(f"OpenAI API Error: {e}")
                            bar.update(1)
                            return
                        await asyncio.sleep(sleep_time)
                        sleep_time *= 2
                _contents.append(resp_2.choices[0].message.parsed.translation)
                _reasons.append(resp_1.choices[0].message.parsed.elaboration)
            self._cur.execute("REPLACE INTO translate(id, content, reason) VALUES (?,?,?);", (
                order,
                json.dumps(_contents, ensure_ascii=False),
                json.dumps(_reasons, ensure_ascii=False)
            ))
            self._db.commit()
            bar.update(1)
