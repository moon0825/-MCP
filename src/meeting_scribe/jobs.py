"""백그라운드 작업 워커. MCP 툴 콜은 즉시 반환하고 전사는 이 스레드가 처리한다.
작업은 SQLite에 영속화 — 서버 재시작 시 미완료 작업 자동 재개."""
import queue
import threading
import uuid
from pathlib import Path

from . import db
from .config import JOBS_DIR
from .pipeline import run_job

_queue: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None


def _loop() -> None:
    while True:
        job_id = _queue.get()
        try:
            run_job(job_id)
        finally:
            _queue.task_done()


def start_worker() -> None:
    global _worker
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_loop, daemon=True, name="scribe-worker")
        _worker.start()
    for job_id in db.pending_jobs():  # 재시작 복구
        _queue.put(job_id)


def enqueue(kind: str, input_path: str, project_id: int | None, options: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    result_dir = JOBS_DIR / job_id
    db.create_job(job_id, kind, str(Path(input_path).resolve()), project_id, options,
                  str(result_dir))
    _queue.put(job_id)
    return job_id
