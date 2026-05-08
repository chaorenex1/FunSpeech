# -*- coding: utf-8 -*-

import asyncio
import threading
import time

from app.core.executor import run_sync_generator


def test_run_sync_generator_respects_bounded_queue_and_yields_all_items():
    async def run():
        def generate():
            for index in range(5):
                yield index

        items = []
        async for item in run_sync_generator(generate, max_queue_size=1):
            items.append(item)
            await asyncio.sleep(0)

        assert items == [0, 1, 2, 3, 4]

    asyncio.run(run())


def test_run_sync_generator_cancel_event_closes_sync_generator():
    async def run():
        closed = threading.Event()

        def generate():
            try:
                index = 0
                while True:
                    yield index
                    index += 1
                    time.sleep(0.01)
            finally:
                closed.set()

        stream = run_sync_generator(generate, max_queue_size=1)
        seen = [await anext(stream), await anext(stream)]
        await stream.aclose()

        assert seen == [0, 1]
        assert closed.wait(timeout=1)

    asyncio.run(run())
