"""Single canonical registry of alert keys.

This module is the ONE place that knows which alerts exist, which family each
belongs to, how it is displayed, which direction it trades, and which config
flag switches it on. It has no imports from the rest of the bot (only the
standard library) so every other module can import it without creating a cycle.

What lives here
    * PIVOT_LEVELS_BUY / PIVOT_LEVELS_SELL
    * FAMILY_PREFIXES
    * TOKEN_NAMES / pretty_alert()
    * AlertSpec / REGISTRY
    * derived views: ALERT_KEYS, BUY_ALERT_KEYS, SELL_ALERT_KEYS,
      ALERT_CONFIG_MAP, ALERT_CONFIG_PREFIX_MAP
    * helpers: alert_family_of, resolve_alert_config_path, keys_for, ...

What does NOT live here
    The condition logic (check_fn / extra_fn) stays in alerts.py.
    validate_alert_definitions() asserts that both sides describe the same keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

__all__ = [
    "PIVOT_LEVELS_BUY", "PIVOT_LEVELS_SELL", "FAMILY_PREFIXES", "TOKEN_NAMES",
    "AlertSpec", "REGISTRY", "ALERT_KEYS", "BUY_ALERT_KEYS", "SELL_ALERT_KEYS",
    "ALERT_CONFIG_MAP", "ALERT_CONFIG_PREFIX_MAP",
    "pretty_alert", "alert_family_of", "resolve_alert_config_path",
    "get_spec", "keys_for", "redis_key_of", "direction_of", "registry_problems",
]

BUY = "buy"
SELL = "sell"

# ── pivot levels (bot_config re-exports these) ───────────────────────────
PIVOT_LEVELS_BUY: List[str] = ["P", "S1", "S2", "S3", "R1", "R2"]
PIVOT_LEVELS_SELL: List[str] = ["P", "S1", "S2", "R1", "R2", "R3"]

# ── family taxonomy (first matching prefix wins) ─────────────────────────
FAMILY_PREFIXES: List[Tuple[str, str]] = [
    ("pivot_", "Pivot"),
    ("strong_reversal_", "Reversal"),
    ("choch_", "Reversal"),
    ("fib_reversal_", "Reversal"),
    ("ob_reversal_", "Order-block"),
    ("vwap_", "VWAP"),
    ("ppo_signal_", "Trend continuation"),
    ("ppo_zero_", "Trend continuation"),
    ("ppo_adaptive_", "Trend continuation"),
    ("ppohist_", "Trend continuation"),
    ("hist_rma_", "Trend continuation"),
    ("rsi_", "Trend continuation"),
    ("cloud_cross_", "Pattern"),
    ("tk_conversion_", "Pattern"),
    ("kijun_cross_", "Pattern"),
    ("equilibrium_cross_", "Pattern"),
    ("dynamic_flow_", "Confluence"),
]

# ── display names ────────────────────────────────────────────────────────
TOKEN_NAMES: Dict[str, str] = {
    "choch": "CHoCH", "ppo": "PPO", "vwap": "VWAP", "rsi": "RSI", "tk": "TK",
    "rma": "RMA", "hist": "Hist", "ppohist": "PPO Hist", "adx": "ADX",
    "macd": "MACD", "fib": "Fib", "up": "UP", "down": "DOWN", "buy": "BUY",
    "sell": "SELL", "s1": "S1", "s2": "S2", "s3": "S3", "r1": "R1", "r2": "R2",
    "r3": "R3", "p": "P", "ob": "OB", "fvg": "FVG", "bos": "BOS", "atr": "ATR",
    "ema": "EMA", "sma": "SMA", "bb": "BB",
}


def pretty_alert(key: str) -> str:
    """strong_reversal_buy -> 'Strong Reversal BUY'."""
    raw = str(key)
    toks = [t for t in raw.split("_") if t and t.lower() != "cross"]
    return " ".join(TOKEN_NAMES.get(t.lower(), t.capitalize()) for t in toks) or raw

def alert_family_of(alert_key: str) -> str:
    """Map a registered alert key to its canonical learning family."""
    key = str(alert_key or "").lower()
    spec = _BY_KEY_LOWER.get(key)

    if spec is not None:
        return spec.family

    return "Other"

def _family_from_prefix(key: str) -> str:
    for prefix, family in FAMILY_PREFIXES:
        if key.startswith(prefix):
            return family
    return "Other"


# ── the registry ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AlertSpec:
    key: str
    family: str
    display_name: str
    direction: str            # "buy" | "sell"
    config_flag: Optional[str]
    kind: str = "standard"    # "standard" | "pivot"

    @property
    def redis_key(self) -> str:
        return f"ALERT:{self.key.upper()}"


# (key, direction, config flag) — keep order stable
_STANDARD_ROWS: List[Tuple[str, str, str, str]] = [
    ("ppo_signal_up", BUY, "Trend continuation", "ENABLE_PPO_ALERTS"),
    ("ppo_signal_down", SELL, "Trend continuation", "ENABLE_PPO_ALERTS"),

    ("rsi_ema5_up", BUY, "Trend continuation", "ENABLE_RSI_ALERTS"),
    ("rsi_ema5_down", SELL, "Trend continuation", "ENABLE_RSI_ALERTS"),

    ("vwap_up", BUY, "VWAP", "ENABLE_VWAP"),
    ("vwap_down", SELL, "VWAP", "ENABLE_VWAP"),

    ("cloud_cross_up", BUY, "Pattern", "ENABLE_CLOUD_CROSS_ALERT"),
    ("cloud_cross_down", SELL, "Pattern", "ENABLE_CLOUD_CROSS_ALERT"),

    ("ob_reversal_buy", BUY, "Order-block", "ENABLE_OB_GATE"),
    ("ob_reversal_sell", SELL, "Order-block", "ENABLE_OB_GATE"),

    ("ppo_zero_up", BUY, "Trend continuation", "ENABLE_PPO_ALERTS"),
    ("ppo_zero_down", SELL, "Trend continuation", "ENABLE_PPO_ALERTS"),

    ("ppo_adaptive_up", BUY, "Trend continuation", "ENABLE_PPO_ALERTS"),
    ("ppo_adaptive_down", SELL, "Trend continuation", "ENABLE_PPO_ALERTS"),

    ("rsi_cross_adaptive_up", BUY, "Trend continuation", "ENABLE_RSI_ALERTS"),
    ("rsi_cross_adaptive_down", SELL, "Trend continuation", "ENABLE_RSI_ALERTS"),

    ("hist_rma_buy", BUY, "Trend continuation", "ENABLE_HIST_RMA"),
    ("hist_rma_sell", SELL, "Trend continuation", "ENABLE_HIST_RMA"),

    ("ppohist_buy", BUY, "Trend continuation", "ENABLE_PPOHIST_ALERT"),
    ("ppohist_sell", SELL, "Trend continuation", "ENABLE_PPOHIST_ALERT"),

    ("tk_conversion_up", BUY, "Pattern", "ENABLE_TK_CONVERSION_CROSS"),
    ("tk_conversion_down", SELL, "Pattern", "ENABLE_TK_CONVERSION_CROSS"),

    ("kijun_cross_up", BUY, "Pattern", "ENABLE_KIJUN_CROSS"),
    ("kijun_cross_down", SELL, "Pattern", "ENABLE_KIJUN_CROSS"),

    ("strong_reversal_buy", BUY, "Reversal", "ENABLE_STRONG_REVERSAL_ALERT"),
    ("strong_reversal_sell", SELL, "Reversal", "ENABLE_STRONG_REVERSAL_ALERT"),

    ("choch_buy", BUY, "Reversal", "ENABLE_CHOCH_ALERT"),
    ("choch_sell", SELL, "Reversal", "ENABLE_CHOCH_ALERT"),

    ("fib_reversal_buy", BUY, "Reversal", "ENABLE_FIB_REVERSAL_ALERT"),
    ("fib_reversal_sell", SELL, "Reversal", "ENABLE_FIB_REVERSAL_ALERT"),

    ("dynamic_flow_cross_buy", BUY, "Confluence", "ENABLE_DYNAMIC_FLOW_CROSS_ALERT"),
    ("dynamic_flow_cross_sell", SELL, "Confluence", "ENABLE_DYNAMIC_FLOW_CROSS_ALERT"),

    ("equilibrium_cross_up", BUY, "Pattern", "ENABLE_EQUILIBRIUM_CROSS"),
    ("equilibrium_cross_down", SELL, "Pattern", "ENABLE_EQUILIBRIUM_CROSS"),
]

def _build() -> Dict[str, AlertSpec]:
    specs: Dict[str, AlertSpec] = {}

    def add(key: str, direction: str, family: str, flag: Optional[str], kind: str) -> None:
        if key in specs:
            raise ValueError(f"alert_registry: duplicate alert key {key!r}")

        specs[key] = AlertSpec(key=key, family=family, display_name=pretty_alert(key), direction=direction, config_flag=flag, kind=kind)

    for key, direction, family, flag in _STANDARD_ROWS:
        add(key, direction, family, flag, "standard")

    for level in PIVOT_LEVELS_BUY:
        add(f"pivot_up_{level}", BUY, "Pivot", "ENABLE_PIVOT", "pivot")

    for level in PIVOT_LEVELS_SELL:
        add(f"pivot_down_{level}", SELL, "Pivot", "ENABLE_PIVOT", "pivot")

    return specs

REGISTRY: Dict[str, AlertSpec] = _build()
_BY_KEY_LOWER: Dict[str, AlertSpec] = {k.lower(): s for k, s in REGISTRY.items()}

# ── derived views ────────────────────────────────────────────────────────
ALERT_KEYS: Dict[str, str] = {k: s.redis_key for k, s in REGISTRY.items()}
BUY_ALERT_KEYS: Set[str] = {k for k, s in REGISTRY.items() if s.direction == BUY}
SELL_ALERT_KEYS: Set[str] = {k for k, s in REGISTRY.items() if s.direction == SELL}

ALERT_CONFIG_MAP: Dict[str, str] = {
    k: s.config_flag
    for k, s in REGISTRY.items()
    if s.config_flag
}

ALERT_CONFIG_PREFIX_MAP: Dict[str, str] = {}

# ── helpers ──────────────────────────────────────────────────────────────
def get_spec(key: str) -> Optional[AlertSpec]:
    return REGISTRY.get(key)


def redis_key_of(key: str) -> Optional[str]:
    spec = REGISTRY.get(key)
    return spec.redis_key if spec else None


def direction_of(key: str) -> Optional[str]:
    spec = REGISTRY.get(key)
    return spec.direction if spec else None


def keys_for(
    *,
    kind: Optional[str] = None,
    family: Optional[str] = None,
    direction: Optional[str] = None,
    config_flag: Optional[str] = None,
) -> List[str]:
    out: List[str] = []
    for k, s in REGISTRY.items():
        if kind is not None and s.kind != kind:
            continue
        if family is not None and s.family != family:
            continue
        if direction is not None and s.direction != direction:
            continue
        if config_flag is not None and s.config_flag != config_flag:
            continue
        out.append(k)
    return out

def resolve_alert_config_path(alert_key: str) -> Optional[str]:
    return ALERT_CONFIG_MAP.get(str(alert_key))


def registry_problems() -> List[str]:
    problems: List[str] = []

    for k, s in REGISTRY.items():
        if not s.family or s.family == "Other":
            problems.append(f"{k}: no canonical family assigned")

        if s.direction not in (BUY, SELL):
            problems.append(f"{k}: bad direction {s.direction!r}")

        if not s.config_flag:
            problems.append(f"{k}: no config flag")

    if BUY_ALERT_KEYS & SELL_ALERT_KEYS:
        problems.append(f"keys in both directions: {sorted(BUY_ALERT_KEYS & SELL_ALERT_KEYS)}")

    return problems