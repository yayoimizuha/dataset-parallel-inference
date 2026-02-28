import argparse
import asyncio
import importlib.util
from asyncio import Semaphore
from importlib.machinery import ModuleSpec
from os import path
from os.path import dirname

import tqdm

from core import InferenceTask


async def main():
    parser = argparse.ArgumentParser(description="Run dataset-parallel-inference project")
    parser.add_argument("--project", type=str, required=True, help="Project name")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrency size")
    args = parser.parse_args()

    task_path = path.join(dirname(__file__), "projects", args.project, "task.py")
    if not path.exists(task_path):
        raise FileNotFoundError(f"Task file not found: {task_path}")

    spec: ModuleSpec = importlib.util.spec_from_file_location("task", task_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # pyright: ignore[reportOptionalMemberAccess]
    task: InferenceTask = module.Task()
    semaphore = Semaphore(value=args.concurrency)
    bar = tqdm.tqdm(total=task.get_length())
    task_queue = set()
    for (order, item) in enumerate(iter(task.dataset)):
        task_queue.add(asyncio.create_task(task.process(item, order, semaphore, bar)))
        if task_queue.__len__() > args.concurrency * 2:
            while task_queue.__len__() > args.concurrency * 1.5:
                _, task_queue = await asyncio.wait(task_queue, timeout=3)
    await asyncio.wait(task_queue)


if __name__ == "__main__":
    asyncio.run(main())
