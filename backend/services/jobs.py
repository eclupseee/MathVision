"""Реестр активных заданий по чатам: даёт /cancel остановить распознавание."""
from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Hashable
from typing import Any, TypeVar

T = TypeVar("T")


class JobCancelled(Exception):
    """Задание отменено пользователем (например, командой /cancel)."""


class ActiveJobs:
    def __init__(self) -> None:
        self._jobs: dict[Hashable, asyncio.Future[Any]] = {}
        self._cancel_requested: set[Hashable] = set()

    def is_active(self, key: Hashable) -> bool:
        job = self._jobs.get(key)
        return job is not None and not job.done()

    def cancel(self, key: Hashable) -> bool:
        job = self._jobs.get(key)
        if job is None or job.done():
            return False
        self._cancel_requested.add(key)
        job.cancel()
        return True


ACTIVE_JOBS = ActiveJobs()


async def run_cancellable(registry: ActiveJobs, key: Hashable, coro: Coroutine[Any, Any, T]) -> T:
    """Выполняет coro как отдельную задачу, которую можно отменить через registry.cancel.
    Отмена пользователем -> JobCancelled; отмена самого обработчика -> CancelledError."""
    job = asyncio.ensure_future(coro)
    registry._jobs[key] = job
    try:
        return await job
    except asyncio.CancelledError:
        if key in registry._cancel_requested:
            raise JobCancelled() from None
        raise
    finally:
        registry._cancel_requested.discard(key)
        if registry._jobs.get(key) is job:
            del registry._jobs[key]
