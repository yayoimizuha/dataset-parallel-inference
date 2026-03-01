from abc import ABC, abstractmethod
from asyncio import Semaphore

import tqdm
from datasets import IterableDataset, Dataset


class InferenceTask(ABC):
    dataset: IterableDataset | Dataset | range | list

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_length(self) -> int:
        pass

    @abstractmethod
    def __del__(self):
        pass

    @abstractmethod
    async def process(self, data, order: int, sem: Semaphore, bar: tqdm.tqdm):
        pass

    async def close(self) -> None:
        """非同期リソースのクリーンアップ。サブクラスで必要に応じてオーバーライドする。"""
        pass
