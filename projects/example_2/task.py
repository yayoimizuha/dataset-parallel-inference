import json
import os
import sqlite3
import asyncio
from os import path
from os.path import dirname
import tqdm
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionUserMessageParam

from core import InferenceTask
from asyncio import Semaphore


class Task(InferenceTask):
    def __init__(self):
        self._db = sqlite3.connect(path.join(dirname(__file__), "db.sqlite"))
        self._cur = self._db.cursor()
        self.dataset = load_dataset("nvidia/Nemotron-Instruction-Following-Chat-v1", streaming=False)["chat_if"]
        self._cur.execute("CREATE TABLE IF NOT EXISTS result(id INT PRIMARY KEY,content TEXT,source TEXT);")
        load_dotenv(path.join(dirname(__file__), ".env"))
        self._client = AsyncOpenAI(api_key=os.environ["API_KEY"], base_url=os.environ["BASE_URL"], timeout=None)

    def get_length(self) -> int:
        return self.dataset.info.splits["chat_if"].num_examples

    def __del__(self):
        self._db.commit()
        self._cur.close()
        self._db.close()

    async def process(self, data, order: int, sem: Semaphore, bar: tqdm.tqdm):
        # id列に order の値が存在するか確認、したらスキップ
        if self._cur.execute("SELECT COUNT(*) FROM result WHERE id=?;", (order,)).fetchone()[0] > 0:
            bar.update(1)
            return
        async with sem:
            input_json = data["messages"]

            original_messages = []
            translated_messages = []

            for message in input_json:
                original_messages.append(message.copy())
                if message["content"] == "":
                    translated_messages.append(message.copy())
                    continue
                sleep_time = 4.0
                while True:
                    try:
                        chat_string = "過去の会話履歴(一貫性のある翻訳のためのコンテキスト):\n\n\n" + \
                            "\n\n\n".join(filter(None,[
                            f"===={orig['role']}=============\n" +
                            (orig['content'] or "") +
                            "\n\n-------↓↓↓↓↓↓-------\n\n" +
                            (trans['content'] or "") +
                            "\n============================="
                            if (orig["content"] or "") != "" else None for orig, trans in
                            zip(original_messages, translated_messages)
                        ])) + "\n\n\n\n" + \
                        "以下に外国語の文章Aが与えられます。その文章を全て日本語に翻訳してください。なお、以下の条件を**遵守**すること。\n" + \
                        "\n" + \
                        " - 人名については翻訳せず、原文での表記のまま書くこと。\n" + \
                        " - 原文に忠実に翻訳し、原文に存在する情報を欠落させたり、書かれていないことを付け加えないこと。\n" + \
                        " - 原文の雰囲気や文脈に基づいて翻訳すること。\n" + \
                        " - 翻訳済みの文章のみを出力し、余計な説明や注釈を加えないこと。\n\n" + \
                        "\n===文章A==========================\n\n\n" + str(message["content"])
                        # print(f"{chat_string}")

                        resp = await self._client.chat.completions.create(
                            messages=[
                            ChatCompletionUserMessageParam(
                                content=chat_string,
                                role="user"
                            )],
                            model=os.environ["MODEL_NAME"],
                            extra_body={"separate_reasoning": True},
                            reasoning_effort="high",
                        )
                        translated_messages.append(resp.choices[0].message.to_dict())
                        break
                    except OpenAIError as e:
                        if sleep_time > 16.0:
                            translated_messages.append({"role": "assistant", "content": "<-- output is missing -->"})
                            break
                        print(f"OpenAI API Error: {e}")
                        await asyncio.sleep(sleep_time)
                        sleep_time *= 2
            # print(json.dumps(output_json, ensure_ascii=False))
            for i in range(input_json.__len__()):
                translated_messages[i].update(role=input_json[i]["role"])
        self._cur.execute("REPLACE INTO result(id, content, source) VALUES (?,?,?);",
            (order,
             json.dumps(translated_messages, ensure_ascii=False),
             json.dumps(input_json.copy(), ensure_ascii=False)),
        )
        self._db.commit()
        bar.update(1)
