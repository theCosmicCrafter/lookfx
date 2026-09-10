"""One worker, one job at a time (single GPU), with progress and cancel."""

from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from lookfx_core.progress import RunContext, Progress, Cancelled


@dataclass
class Job:
    id: str
    kind: str                       # "render" | "solve" | "proxy"
    state: str = "queued"           # queued | running | done | failed | cancelled
    stage: str = ""
    done: int = 0
    total: int = 0
    message: str = ""
    started: float | None = None
    finished: float | None = None
    result: Any = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    _listeners: list[queue.Queue] = field(default_factory=list)

    @property
    def eta_s(self) -> float | None:
        if self.state != "running" or not self.started or self.done <= 0 or self.total <= 0:
            return None
        elapsed = time.time() - self.started
        return elapsed * (self.total - self.done) / self.done

    def to_json(self) -> dict:
        return {"id": self.id, "kind": self.kind, "state": self.state, "stage": self.stage,
                "done": self.done, "total": self.total, "message": self.message,
                "eta_s": self.eta_s, "error": self.error,
                "result": self.result if isinstance(self.result, (dict, list, str, int, float, type(None))) else None}

    def _emit(self, event: str, data: dict) -> None:
        for q in list(self._listeners):
            q.put((event, data))

    def listen(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        self._listeners.append(q)
        return q

    def unlisten(self, q: queue.Queue) -> None:
        try:
            self._listeners.remove(q)
        except ValueError:
            pass


class JobRunner:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="lookfx-jobs")
        self._thread.start()

    def submit(self, kind: str, fn: Callable[[RunContext], Any], ctx: RunContext | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        ctx = ctx or RunContext()
        ctx.cancel = job.cancel

        def on_progress(p: Progress):
            job.stage, job.done, job.total, job.message = p.stage, p.done, p.total, p.message
            job._emit("progress", job.to_json())
        ctx.on_progress = on_progress
        with self._lock:
            self.jobs[job.id] = job
        self._q.put((job, fn, ctx))
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        job.cancel.set()
        if job.state == "queued":
            job.state = "cancelled"
            job._emit("done", job.to_json())
        return True

    def _loop(self):
        while True:
            job, fn, ctx = self._q.get()
            if job.cancel.is_set():
                job.state = "cancelled"
                job._emit("done", job.to_json())
                continue
            job.state, job.started = "running", time.time()
            job._emit("progress", job.to_json())
            try:
                job.result = fn(ctx)
                job.state = "done"
            except Cancelled:
                job.state = "cancelled"
            except Exception as e:  # noqa: BLE001
                job.state, job.error = "failed", f"{type(e).__name__}: {e}"
                traceback.print_exc()
            job.finished = time.time()
            job._emit("done", job.to_json())
