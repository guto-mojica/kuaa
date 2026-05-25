"""Unit tests for the JobState multi-subscriber broadcaster + log buffer.

Background
----------
The Processing tab previously routed pipeline progress through a
single-consumer ``queue.Queue`` on ``JobState``. That made the SSE
stream's "user opens a second tab" / "user reloads while a job runs"
paths fundamentally racy — each event went to whichever consumer
called ``get_nowait`` first, so other consumers saw half the stream.

This module pins the contract of the replacement: a pub/sub
broadcaster where every subscriber sees every event, plus a bounded
log ring buffer that survives page navigation so a returning user can
replay the captured pipeline output.

Hermetic: nothing here touches the disk, the FastAPI app, or a real
pipeline. We construct ``JobState`` directly and exercise the
broadcaster/log surface in isolation.
"""

from __future__ import annotations

import queue
import threading

import pytest

# ── EventBroadcaster ──────────────────────────────────────────────────────────


def test_broadcaster_delivers_event_to_every_subscriber():
    """Every active subscriber sees every published event.

    The previous queue.Queue model would deliver each event to ONLY
    ONE consumer (whichever called ``get_nowait`` first). The pub/sub
    broadcaster MUST fan out to all currently-subscribed queues, so a
    user with the Processing tab open in two browser tabs sees the
    same live stream in both.
    """
    from api.jobs import EventBroadcaster

    bus = EventBroadcaster()
    q1 = bus.subscribe()
    q2 = bus.subscribe()

    bus.publish(("update", None))

    assert q1.get_nowait() == ("update", None)
    assert q2.get_nowait() == ("update", None)


def test_broadcaster_unsubscribe_stops_delivery():
    """Unsubscribing removes the queue from the fan-out set.

    SSE generators MUST unsubscribe when the connection closes so a
    dead client's queue doesn't keep accumulating events for the rest
    of the job's lifetime.
    """
    from api.jobs import EventBroadcaster

    bus = EventBroadcaster()
    q = bus.subscribe()
    bus.unsubscribe(q)

    bus.publish(("update", None))

    with pytest.raises(queue.Empty):
        q.get_nowait()


def test_broadcaster_publish_with_no_subscribers_does_not_raise():
    """publish() before any subscribe() must be a silent no-op.

    The pipeline runner publishes ``log`` events from the worker
    thread before the SSE generator has registered. Those events are
    lost (no buffer in the broadcaster itself), which is intentional:
    the log BUFFER on JobState is the persistence layer; the
    broadcaster is the live wire.
    """
    from api.jobs import EventBroadcaster

    bus = EventBroadcaster()
    # Should not raise:
    bus.publish(("update", None))


def test_broadcaster_drops_event_on_full_subscriber_without_blocking_others():
    """A slow/dead subscriber whose queue is full MUST NOT block the
    producer or starve other subscribers.

    The worker thread publishes events at pipeline speed; if one SSE
    consumer is stuck (network back-pressure, paused tab), we drop
    events to its queue rather than block the worker. Other
    subscribers keep receiving cleanly.
    """
    from api.jobs import EventBroadcaster

    bus = EventBroadcaster()
    slow = bus.subscribe(maxsize=1)
    fast = bus.subscribe(maxsize=10)
    slow.put_nowait(("seed", None))  # fill slow's queue

    # This publish should NOT raise (slow drops, fast accepts):
    bus.publish(("update", None))

    assert fast.get_nowait() == ("update", None)
    # slow still only has its seed entry:
    assert slow.get_nowait() == ("seed", None)
    with pytest.raises(queue.Empty):
        slow.get_nowait()


def test_broadcaster_is_thread_safe_under_concurrent_publish():
    """Concurrent publishes from many threads MUST NOT lose or
    duplicate events for any subscriber.

    The pipeline runner is single-threaded today, but the log handler
    can be invoked from any logger — including background threads
    inside cinemateca.* modules — so the broadcaster's subscriber
    list must be safe to iterate under concurrent publish().
    """
    from api.jobs import EventBroadcaster

    bus = EventBroadcaster()
    q = bus.subscribe(maxsize=10_000)

    threads = [
        threading.Thread(target=lambda i=i: [bus.publish(("update", i)) for _ in range(100)])
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    drained = []
    while True:
        try:
            drained.append(q.get_nowait())
        except queue.Empty:
            break

    assert len(drained) == 8 * 100, f"expected 800 events, got {len(drained)}"


# ── JobState integration ──────────────────────────────────────────────────────


def test_jobstate_has_bounded_log_deque():
    """JobState carries a ring buffer for captured pipeline log lines.

    The buffer is what survives page navigation: returning users
    replay it on /tab/processing and on SSE reconnect. Bounded so a
    long-running pipeline can't blow up memory.
    """
    from api.jobs import LOG_BUFFER_MAXLEN, JobState

    job = JobState(id="j1", video_path="x.mp4")

    assert job.log.maxlen == LOG_BUFFER_MAXLEN
    assert len(job.log) == 0

    for i in range(LOG_BUFFER_MAXLEN + 5):
        job.log.append({"t": "00:00:00", "lv": "i", "m": f"row {i}"})

    assert len(job.log) == LOG_BUFFER_MAXLEN
    # The oldest 5 rows were evicted; the newest row is at the right:
    assert job.log[-1]["m"] == f"row {LOG_BUFFER_MAXLEN + 4}"


def test_jobstate_publish_routes_through_broadcaster():
    """job.publish(name, data) MUST fan out via the broadcaster.

    This is the surface the pipeline runner uses; tests pin it
    against direct broadcaster access so any future "publish() also
    writes to a log file" extension shows up cleanly here.
    """
    from api.jobs import JobState

    job = JobState(id="j1", video_path="x.mp4")
    q = job.subscribe()

    job.publish("update")
    job.publish("log", {"t": "00:00:01", "lv": "i", "m": "started step"})

    assert q.get_nowait() == ("update", None)
    assert q.get_nowait() == ("log", {"t": "00:00:01", "lv": "i", "m": "started step"})


def test_jobstate_unsubscribe_returns_queue_to_idle():
    """job.unsubscribe(q) MUST stop further deliveries to q.

    Required for SSE generator cleanup so a closed connection's queue
    doesn't grow for the rest of the job's lifetime.
    """
    from api.jobs import JobState

    job = JobState(id="j1", video_path="x.mp4")
    q = job.subscribe()
    job.unsubscribe(q)

    job.publish("update")

    with pytest.raises(queue.Empty):
        q.get_nowait()
