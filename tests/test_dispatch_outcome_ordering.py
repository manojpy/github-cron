"""Behavioural tests: outcome + ACTIVE state must follow Telegram delivery.

A failed send must NOT leave a recorded trade or an ACTIVE alert state behind
(phantom trade), and a failing post-send hook must not skip the
candle-processed marker (which would allow a duplicate alert next run).
"""
import asyncio
import logging

from alerts import AlertPayload, dispatch_combined_alerts


class FakeTelegram:
    def __init__(self, results):
        self.results = list(results)   # one bool per send() call
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)
        return self.results.pop(0) if self.results else False

class FakeSdb:
    def __init__(self):
        self.state_batches = []
        self.processed = []
        self.released = []

    async def atomic_batch_update(self, changes):
        self.state_batches.append(list(changes))
        return True

    async def set_last_processed_candle_ts(self, pair, ts):
        self.processed.append((pair, ts))

    async def release_recent_alert(self, pair, key):
        self.released.append((pair, key))


def _payload(pair, calls, *, record_raises=False):
    async def record():
        calls.append(pair)
        if record_raises:
            raise RuntimeError("redis down")

    return AlertPayload(
        pair_name=pair, direction="buy", score=5.0, total=10.0,
        msg_body=f"body {pair}", dedup_keys=["k"],
        state_changes=[(f"{pair}:state", "ACTIVE", None)],
        budget_count=1, ts=1_700_000_000,
        alert_keys=["ppo_cross_up"],
        record_win_rate=record, mark_candle_processed=True,
    )


def _run(payloads, telegram, sdb):
    return asyncio.run(dispatch_combined_alerts(
        payloads, telegram, None, sdb, [0], asyncio.Lock(), 50,
        logging.getLogger("test"),
    ))


def test_send_success_records_and_activates_once():
    calls, sdb = [], FakeSdb()
    n = _run([_payload("ETHUSD", calls)], FakeTelegram([True]), sdb)
    assert n == 1
    assert calls == ["ETHUSD"]
    assert sdb.state_batches == [[("ETHUSD:state", "ACTIVE", None)]]
    assert sdb.processed == [("ETHUSD", 1_700_000_000)]


def test_send_failure_creates_no_phantom_trade():
    calls, sdb = [], FakeSdb()
    # combined send fails, individual fallback send fails too
    n = _run([_payload("ETHUSD", calls)], FakeTelegram([False, False]), sdb)
    assert n == 0
    assert calls == []                 # no outcome recorded
    assert sdb.state_batches == []     # alert state NOT activated
    assert sdb.processed == []


def test_post_send_hook_error_still_marks_candle_processed():
    calls, sdb = [], FakeSdb()
    n = _run([_payload("ETHUSD", calls, record_raises=True)], FakeTelegram([True]), sdb)
    assert n == 1
    assert calls == ["ETHUSD"]
    assert sdb.processed == [("ETHUSD", 1_700_000_000)]