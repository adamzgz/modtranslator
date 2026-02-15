"""Background worker thread for translation tasks.

Runs pipeline functions in a separate thread and communicates progress
back to the GUI main thread via a queue.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from queue import Queue
from threading import Event, Thread
from typing import Any, Callable


@dataclass
class WorkerMessage:
    """Message from worker thread to GUI."""
    type: str  # "progress", "log", "done", "error"
    phase: str = ""
    current: int = 0
    total: int = 0
    message: str = ""
    result: Any = None


class TranslationWorker:
    """Manages a single background translation thread."""

    def __init__(self) -> None:
        self._thread: Thread | None = None
        self._cancel_event = Event()
        self.queue: Queue[WorkerMessage] = Queue()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, task_fn: Callable[..., Any], **kwargs: Any) -> None:
        """Launch a task function in a background thread.

        The task_fn should accept on_progress and cancel_event kwargs.
        """
        if self.is_running:
            return

        self._cancel_event.clear()
        # Drain any old messages
        while not self.queue.empty():
            self.queue.get_nowait()

        kwargs["on_progress"] = self._on_progress
        kwargs["cancel_event"] = self._cancel_event

        self._thread = Thread(target=self._run, args=(task_fn,), kwargs=kwargs, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Signal the worker to stop."""
        self._cancel_event.set()

    def _on_progress(self, phase: str, current: int, total: int, message: str = "") -> None:
        """Progress callback invoked from the worker thread."""
        self.queue.put(WorkerMessage(
            type="progress",
            phase=phase,
            current=current,
            total=total,
            message=message,
        ))

    def _run(self, task_fn: Callable[..., Any], **kwargs: Any) -> None:
        """Execute the task and put done/error message on queue."""
        try:
            result = task_fn(**kwargs)
            self.queue.put(WorkerMessage(type="done", result=result))
        except Exception as e:
            self.queue.put(WorkerMessage(
                type="error",
                message=f"{e}\n{traceback.format_exc()}",
            ))
