import json
import os
import sqlite3
from os import path
from os.path import dirname
import tqdm
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from core import InferenceTask
from asyncio import Semaphore


class Task(InferenceTask):

    def __init__(self):
        self._db = sqlite3.connect(path.join(dirname(__file__), "db.sqlite"))
        self._cur = self._db.cursor()
        self.dataset = iter(load_dataset("nvidia/Nemotron-Instruction-Following-Chat-v1", streaming=True)["chat_if"])
        self._cur.execute('CREATE TABLE IF NOT EXISTS result(id INT PRIMARY KEY,content TEXT);')
        load_dotenv(path.join(dirname(__file__), ".env"))
        self._client = AsyncOpenAI(
            api_key=os.environ["API_KEY"],
            base_url=os.environ["BASE_URL"]
        )

    def get_length(self) -> int:
        return self.dataset.info.splits["chat_if"].num_examples

    def __del__(self):
        self._db.commit()
        self._cur.close()
        self._db.close()

    async def process(self, data, order: int, sem: Semaphore, bar: tqdm.tqdm):
        async with sem:
            input_json = data["messages"]
            output_json = [
                {
                    "role": "system",
                    "content": "外国語の文章と、日本語の文章のペアが与えられるので、外国語の文章を日本語に翻訳してください。なお、以下の条件を守ってください。\n"
                               " - 人名は翻訳せず、原文の表記のまま用いること。\n"
                               " - 原文に忠実に翻訳し、原文に存在する情報を欠落させたり、原文に書いていないことを勝手に付け加えないこと。\n"
                               " - 原文の雰囲気・文脈に基づいて翻訳すること。"
                }
            ]
            for message in input_json:
                output_json.append(message)
                output_json[-1].update(role="user")
                if output_json[-1]["content"] == "":
                    output_json.append({"role": "assistant", "content": ""})
                    continue
                resp = await self._client.chat.completions.create(
                    messages=output_json,
                    model=os.environ["MODEL_NAME"]
                )
                output_json.append(resp.choices[0].message.to_dict())
            # print(json.dumps(output_json, ensure_ascii=False))
            output_json = output_json[2::2]
            for i in range(input_json.__len__()):
                output_json[i].update(role=input_json[i]["role"])
        self._cur.execute("REPLACE INTO result(id, content) VALUES (?,?);",
                          (order, json.dumps(output_json, ensure_ascii=False)))
        self._db.commit()
        bar.update(1)
