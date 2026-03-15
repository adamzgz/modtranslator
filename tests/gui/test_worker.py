"""Tests for GUI background worker thread."""

from __future__ import annotations

import time
from threading import Event

from modtranslator.gui.worker import TranslationWorker, WorkerMessage

# ---------------------------------------------------------------------------
# WorkerMessage dataclass
# ---------------------------------------------------------------------------


class TestWorkerMessage:
    def test_defaults(self) -> None:
        msg = WorkerMessage(type="progress")
        assert msg.type == "progress"
        assert msg.phase == ""
        assert msg.current == 0
        assert msg.total == 0
        assert msg.message == ""
        assert msg.result is None

    def test_custom_values(self) -> None:
        msg = WorkerMessage(
            type="done", phase="write", current=5, total=10,
            message="ok", result={"key": "val"},
        )
        assert msg.type == "done"
        assert msg.phase == "write"
        assert msg.current == 5
        assert msg.total == 10
        assert msg.message == "ok"
        assert msg.result == {"key": "val"}


# ---------------------------------------------------------------------------
# TranslationWorker
# ---------------------------------------------------------------------------


class TestTranslationWorker:
    def test_is_running_initially_false(self) -> None:
        w = TranslationWorker()
        assert w.is_running is False

    def test_start_launches_thread(self) -> None:
        w = TranslationWorker()

        def task(on_progress: object, cancel_event: object) -> str:
            return "done"

        w.start(task)
        assert w._thread is not None
        w._thread.join(timeout=2)
        assert w.is_running is False

    def test_start_injects_kwargs(self) -> None:
        """start() should inject on_progress and cancel_event into kwargs."""
        captured: dict = {}

        def task(**kwargs: object) -> None:
            captured.update(kwargs)

        w = TranslationWorker()
        w.start(task)
        w._thread.join(timeout=2)  # type: ignore[union-attr]

        assert "on_progress" in captured
        assert "cancel_event" in captured
        assert callable(captured["on_progress"])
        assert isinstance(captured["cancel_event"], Event)

    def test_start_passes_extra_kwargs(self) -> None:
        captured: dict = {}

        def task(**kwargs: object) -> None:
            captured.update(kwargs)

        w = TranslationWorker()
        w.start(task, lang="ES", game="fo3")
        w._thread.join(timeout=2)  # type: ignore[union-attr]

        assert captured["lang"] == "ES"
        assert captured["game"] == "fo3"

    def test_start_while_running_is_noop(self) -> None:
        """Starting a second task while one is running should be a no-op."""
        barrier = Event()

        def slow_task(on_progress: object, cancel_event: object) -> None:
            barrier.wait(timeout=5)

        w = TranslationWorker()
        w.start(slow_task)
        first_thread = w._thread

        # Try to start again — should be ignored
        w.start(slow_task)
        assert w._thread is first_thread

        barrier.set()
        first_thread.join(timeout=2)  # type: ignore[union-attr]

    def test_cancel_sets_event(self) -> None:
        w = TranslationWorker()
        assert not w._cancel_event.is_set()
        w.cancel()
        assert w._cancel_event.is_set()

    def test_start_clears_cancel_event(self) -> None:
        w = TranslationWorker()
        w.cancel()
        assert w._cancel_event.is_set()

        def task(on_progress: object, cancel_event: object) -> None:
            pass

        w.start(task)
        # cancel_event should have been cleared before the task ran
        assert not w._cancel_event.is_set()
        w._thread.join(timeout=2)  # type: ignore[union-attr]

    def test_done_message_on_success(self) -> None:
        w = TranslationWorker()

        def task(on_progress: object, cancel_event: object) -> str:
            return "result_value"

        w.start(task)
        w._thread.join(timeout=2)  # type: ignore[union-attr]

        msg = w.queue.get(timeout=1)
        assert msg.type == "done"
        assert msg.result == "result_value"

    def test_error_message_on_exception(self) -> None:
        w = TranslationWorker()

        def bad_task(on_progress: object, cancel_event: object) -> None:
            raise ValueError("boom")

        w.start(bad_task)
        w._thread.join(timeout=2)  # type: ignore[union-attr]

        msg = w.queue.get(timeout=1)
        assert msg.type == "error"
        assert "boom" in msg.message

    def test_progress_messages_in_queue(self) -> None:
        w = TranslationWorker()

        def task(on_progress: object, cancel_event: object) -> None:
            on_progress("scan", 1, 10, "scanning...")  # type: ignore[operator]
            on_progress("translate", 5, 10, "translating...")  # type: ignore[operator]

        w.start(task)
        w._thread.join(timeout=2)  # type: ignore[union-attr]

        msgs = []
        while not w.queue.empty():
            msgs.append(w.queue.get_nowait())

        progress_msgs = [m for m in msgs if m.type == "progress"]
        assert len(progress_msgs) == 2
        assert progress_msgs[0].phase == "scan"
        assert progress_msgs[0].current == 1
        assert progress_msgs[0].total == 10
        assert progress_msgs[0].message == "scanning..."
        assert progress_msgs[1].phase == "translate"

        done_msgs = [m for m in msgs if m.type == "done"]
        assert len(done_msgs) == 1

    def test_cooperative_cancellation(self) -> None:
        """Task that checks cancel_event should stop early."""
        w = TranslationWorker()
        iterations = 0

        def cancellable_task(on_progress: object, cancel_event: Event) -> str:
            nonlocal iterations
            for _i in range(100):
                if cancel_event.is_set():
                    return "cancelled"
                iterations += 1
                time.sleep(0.01)
            return "completed"

        w.start(cancellable_task)
        time.sleep(0.05)
        w.cancel()
        w._thread.join(timeout=2)  # type: ignore[union-attr]

        msg = w.queue.get(timeout=1)
        assert msg.type == "done"
        assert msg.result == "cancelled"
        assert iterations < 100

    def test_start_drains_old_messages(self) -> None:
        """Starting a new task should drain leftover messages from the queue."""
        w = TranslationWorker()
        w.queue.put(WorkerMessage(type="done", message="stale"))

        def task(on_progress: object, cancel_event: object) -> str:
            return "fresh"

        w.start(task)
        w._thread.join(timeout=2)  # type: ignore[union-attr]

        msgs = []
        while not w.queue.empty():
            msgs.append(w.queue.get_nowait())

        assert all(m.message != "stale" for m in msgs)
        assert any(m.type == "done" and m.result == "fresh" for m in msgs)

    def test_is_running_true_while_thread_alive(self) -> None:
        barrier = Event()

        def blocking_task(on_progress: object, cancel_event: object) -> None:
            barrier.wait(timeout=5)

        w = TranslationWorker()
        w.start(blocking_task)
        assert w.is_running is True

        barrier.set()
        w._thread.join(timeout=2)  # type: ignore[union-attr]
        assert w.is_running is False
