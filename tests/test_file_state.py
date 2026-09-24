"""Round-trip tests for the file-backed Redis adapter.

These verify the one property the whole state migration rests on: what a
process writes before close() is readable by a fresh adapter constructed
against the same data directory. If this breaks, production symptoms are
"dedup stopped working" or "alerts re-fire every run" — visible only after
15 minutes and easy to misattribute. This catches it in milliseconds.
"""
from __future__ import annotations

import asyncio
import json
import time


def test_file_state_survives_restart(tmp_path):
    """Every storage type (kv, hash, list, stream) plus TTLs persist across
    a simulated restart: close() flushes, a fresh adapter re-reads."""
    from file_state import _FileRedisAdapter

    async def scenario():
        # ── First process: write, then flush on close ──
        first = _FileRedisAdapter(str(tmp_path))
        await first.connect()

        await first.set(
            "pair_state:BTCUSD",
            json.dumps({
                "state": "ACTIVE",
                "ts": 1234567890,
            }),
        )

        await first.hset(
            "alert_stats:BTCUSD",
            mapping={
                "wins": "12",
                "losses": "8",
            },
        )

        await first.lpush(
            "history:BTCUSD",
            "100",
        )

        await first.xadd(
            "outcome_log_stream",
            {
                "pair": "BTCUSD",
                "win": "1",
            },
        )

        # TTL'd key that should still be alive across the restart.
        await first.set("dedup_key", "1", ex=3600)

        # TTL'd key that expires before the second adapter reads it.
        await first.set("expired_key", "1", ex=1)

        await first.close()

        # ── Second process: fresh adapter, same data dir ──
        second = _FileRedisAdapter(str(tmp_path))
        await second.connect()

        # kv
        assert await second.get("pair_state:BTCUSD") is not None

        # hash
        assert await second.hgetall(
            "alert_stats:BTCUSD"
        ) == {
            "wins": "12",
            "losses": "8",
        }

        # list
        assert await second.lrange(
            "history:BTCUSD",
            0,
            -1,
        ) == ["100"]

        # stream
        stream = await second.xrevrange(
            "outcome_log_stream",
            count=1,
        )
        assert len(stream) == 1
        assert stream[0][1]["pair"] == "BTCUSD"

        # TTL sidecar: dedup_key survived with its expiry intact.
        assert await second.get("dedup_key") == "1"
        assert "dedup_key" in second._ttls

        # TTL sidecar: expired_key was pruned on read by _expired().
        # Sleep past the 1-second expiry before the read.
        time.sleep(2)
        assert await second.get("expired_key") is None

        await second.close()

    asyncio.run(scenario())