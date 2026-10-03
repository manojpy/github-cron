"""Guards for standing-decision TTLs and the alert-key registry.

1. Every durable metadata key gets the long TTL (not the 7-day default) and
   the set stays identical to redis_audit.DURABLE_METADATA.
2. Every alert key maps to a real family (none fall into "Other"), so a newly
   added alert cannot silently skip family analysis.
"""
import asyncio

import redis_audit
import state
from alerts import ALERT_KEYS
from threshold_engine import alert_family_of


class _Store:
    """Just enough of RedisStateStore for set_metadata()."""

    meta_prefix = "metadata:"
    metadata_expiry_seconds = 7 * 86400
    set_metadata = state.RedisStateStore.set_metadata

    def __init__(self):
        self.calls = []

    async def _safe_redis_op(self, op, timeout, name, **_kw):
        class _R:
            async def set(_s, key, value, ex=None):
                self.calls.append((key, ex))

        import state as _state
        orig = _state._rc
        _state._rc = lambda _r: _R()
        try:
            return await op()
        finally:
            _state._rc = orig

    _redis = object()


def _ttl_for(key, **kw):
    store = _Store()
    asyncio.run(store.set_metadata(key, "{}", **kw))
    return store.calls[0][1]


def test_durable_keys_get_long_ttl():
    for key in state.DURABLE_METADATA_KEYS:
        assert _ttl_for(key) == state.DURABLE_METADATA_TTL_SEC


def test_non_durable_key_keeps_default_ttl():
    assert _ttl_for("daily_cache:BTCUSD:2026-10-01") == 7 * 86400


def test_explicit_ttl_wins():
    assert _ttl_for("dynamic_weights", ttl=123) == 123


def test_durable_sets_match_audit():
    assert set(state.DURABLE_METADATA_KEYS) == set(redis_audit.DURABLE_METADATA)


def test_alert_registry_is_self_consistent():
    from alert_registry import (
        REGISTRY,
        ALERT_KEYS,
        BUY_ALERT_KEYS,
        SELL_ALERT_KEYS,
        alert_family_of,
        registry_problems,
    )

    assert REGISTRY
    assert not registry_problems()

    assert set(ALERT_KEYS) == set(REGISTRY)

    assert not (BUY_ALERT_KEYS & SELL_ALERT_KEYS)

    for key, spec in REGISTRY.items():
        assert spec.key == key
        assert spec.family
        assert spec.family != "Other"
        assert spec.direction in {"buy", "sell"}
        assert spec.config_flag
        assert alert_family_of(key) == spec.family

def test_unknown_alert_key_does_not_inherit_family():
    from alert_registry import alert_family_of

    assert alert_family_of("vwap_future") == "Other"
    assert alert_family_of("ppo_unknown") == "Other"
    assert alert_family_of("rsi_unknown") == "Other"
    assert alert_family_of("not_registered") == "Other"

def test_all_alert_keys_have_a_family():
    other = [k for k in ALERT_KEYS if alert_family_of(k) == "Other"]
    assert not other, f"alert keys with no family: {other}"
