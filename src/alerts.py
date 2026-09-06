from __future__ import annotations
import time
import re
import json
import random
import asyncio
import logging
from enum import StrEnum
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Set, Callable, Union

import numpy as np

from bot_config import (
    cfg, logger, Constants, CompiledPatterns, PIVOT_LEVELS_BUY, PIVOT_LEVELS_SELL,
    shutdown_event, format_ist_time, json_dumps, CONFLUENCE_WEIGHTS, BtcMacroContext,
    ClusterContext, _get_session_from_ts,
)
from fetcher import (
    PriceData, DataFetcher, SessionManager, compute_backoff, validate_indicator_values,
    CandleSnapshot, independent_candle_reverify, cross_check_15m_against_5m,
    confirm_candle_unchanged, detect_reversal_candle_pattern,
)
from state import RedisStateStore, TokenBucket
from gates import GateResult, IndicatorCache
import threshold_engine as engine

from indicators import (
    calculate_alert_indicators_numpy, validate_indicators_dict, validate_vwap_cross,
    validate_cloud_cross, validate_conversion_cross,
    _fib_reversal_confluence_vote,
)
from threshold_engine import hash_config_state

# Runtime override for CONFLUENCE_WEIGHTS (loaded from Redis at startup)
# Falls back to static cfg.CONFLUENCE_WEIGHTS if not set
RUNTIME_CONFLUENCE_WEIGHTS: Dict[str, float] = dict(CONFLUENCE_WEIGHTS)

def escape_markdown_v2(text: str) -> str:
    return CompiledPatterns.ESCAPE_MARKDOWN.sub(r'\\\g<0>', str(text))

class TelegramQueue:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.token_bucket = TokenBucket(cfg.TELEGRAM_RATE_LIMIT_PER_MINUTE, cfg.TELEGRAM_BURST_SIZE)

    async def send(self, message: str, priority: str = "normal") -> bool:
        try:
            return bool(
                await asyncio.wait_for(
                    self._send_impl(message),
                    timeout=45.0
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            if cfg.FAIL_ON_TELEGRAM_DOWN:
                raise
            return False

    async def _send_impl(self, message: str) -> bool:
        await self.token_bucket.acquire()
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        params = {"chat_id": self.chat_id, "text": message, "parse_mode": "MarkdownV2"}
        session = await SessionManager.get_session()
        for attempt in range(1, cfg.TELEGRAM_RETRIES + 1):
            if shutdown_event.is_set():
                return False
            try:
                async with session.post(url, data=params, timeout=10) as resp:
                    if resp.status == 429:
                        wait_sec = min(int(resp.headers.get("Retry-After", 1)), Constants.CIRCUIT_BREAKER_MAX_WAIT)
                        await asyncio.sleep(wait_sec + random.uniform(0.1, 0.5))
                        continue
                    if resp.status == 200:
                        return True
                    if resp.status in (400, 401, 403, 404):
                        logger.error(f"Telegram API error {resp.status} - check token/chat_id")
                        return False
                    raise Exception(f"Telegram API error {resp.status}")

            except Exception as e:
                logger.warning(f"Telegram send attempt {attempt} failed: {e}")
                if attempt < cfg.TELEGRAM_RETRIES:
                    delay = compute_backoff(1.0, attempt)
                    logger.debug(f"Retrying Telegram request in {delay:.1f}s (attempt {attempt})...")
                    await asyncio.sleep(delay)
        return False

    async def send_with_markup(self, message: str, reply_markup: Dict[str, Any]) -> Optional[int]:
        """Like send(), but attaches an inline keyboard and returns the sent
        message_id (needed later to edit the buttons off) instead of a bool.
        Kept as a separate method rather than changing send()'s return type,
        since every existing call site depends on send() returning bool.
        Returns None on failure — treat the same as a failed send()."""
        try:
            return await asyncio.wait_for(
                self._send_impl_with_markup(message, reply_markup), timeout=45.0
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Telegram send_with_markup failed: {e}")
            if cfg.FAIL_ON_TELEGRAM_DOWN:
                raise
            return None

    async def _send_impl_with_markup(self, message: str, reply_markup: Dict[str, Any]) -> Optional[int]:
        await self.token_bucket.acquire()
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        params = {
            "chat_id": self.chat_id, "text": message, "parse_mode": "MarkdownV2",
            "reply_markup": json_dumps(reply_markup),
        }
        session = await SessionManager.get_session()
        for attempt in range(1, cfg.TELEGRAM_RETRIES + 1):
            if shutdown_event.is_set():
                return None
            try:
                async with session.post(url, data=params, timeout=10) as resp:
                    if resp.status == 429:
                        wait_sec = min(int(resp.headers.get("Retry-After", 1)), Constants.CIRCUIT_BREAKER_MAX_WAIT)
                        await asyncio.sleep(wait_sec + random.uniform(0.1, 0.5))
                        continue
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("result", {}).get("message_id")
                    if resp.status in (400, 401, 403, 404):
                        response_text = await resp.text()
                        try:
                            response_json = json.loads(response_text)
                            description = response_json.get("description", response_text)
                        except Exception:
                            description = response_text
                        
                        logger.error(
                            f"Telegram API error {resp.status}: {description} | "
                            f"chat_id={getattr(self, 'chat_id', '?')}"
                        )
                        return None
                    raise Exception(f"Telegram API error {resp.status}")

            except Exception as e:
                logger.warning(f"Telegram send_with_markup attempt {attempt} failed: {e}")
                if attempt < cfg.TELEGRAM_RETRIES:
                    delay = compute_backoff(1.0, attempt)
                    logger.debug(f"Retrying Telegram request in {delay:.1f}s (attempt {attempt})...")
                    await asyncio.sleep(delay)
        return None

def _clean_extra_text(extra: Optional[str]) -> str:
    """Helper to strip emojis, OHLC data, and technical metadata."""
    if not extra:
        return ""
    extra_clean = re.sub(r'[🟢🔴🔵🟣]', '', extra)  
    extra_clean = re.sub(r'\(O:[\d.]+ H:[\d.]+ L:[\d.]+ C:[\d.]+\)', '', extra_clean)  
    extra_clean = re.sub(r'\[i15=\d+,\s*[\d-]+\s+[\d:]+\s+IST\]', '', extra_clean)  
    return extra_clean.strip()

def _format_price(price: Any) -> str:
    """Safely format price to 2 decimal places."""
    return f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A"

def _fmt_num(n: float) -> str:
    """Format a number with no trailing '.0' when it's a whole number."""
    return f"{n:.0f}" if float(n).is_integer() else f"{n:.1f}"

def _fmt_score(score: Optional[float], total: Optional[float] = None) -> str:
    """Compact weighted-confluence-score suffix for message headers.
    e.g. ' - 88%(26.5/30)' when total is known, else '(6.5)' as a fallback.
    Returned as raw (unescaped) text; caller is responsible for MarkdownV2 escaping."""
    if score is None:
        return ""
    if total is not None and total > 0:
        pct = round((score / total) * 100)
        return f" - {pct}%({_fmt_num(score)}/{_fmt_num(total)})"
    return f"({_fmt_num(score)})"

def build_single_msg(title: str, pair: str, price: Any, ts: int, extra: Optional[str] = None, score: Optional[float] = None, total: Optional[float] = None) -> str:
    if not title: 
        title = "ALERT"
    
    parts = title.split(" ", 1)
    symbols = parts[0]
    description = parts[1] if len(parts) == 2 else title
    
    # 1. Format the raw strings
    price_str = _format_price(price)
    extra_clean = _clean_extra_text(extra)
    date_str = format_ist_time(ts, '%d-%m-%Y')
    time_str = format_ist_time(ts, '%H:%M IST')
    
    # 2. ESCAPE INDIVIDUAL DATA (Crucial for MarkdownV2 stability)
    e_symbols = escape_markdown_v2(symbols)
    e_pair = escape_markdown_v2(pair)
    e_score = escape_markdown_v2(_fmt_score(score, total)) 
    e_price = escape_markdown_v2(price_str)
    e_desc = escape_markdown_v2(description)
    e_extra = escape_markdown_v2(extra_clean)
    e_date = escape_markdown_v2(date_str)
    e_time = escape_markdown_v2(time_str)
    
    line1 = f"{e_symbols} *{e_pair}{e_score}* \\- *{e_price}*"

    # Bold the alert type, italicize the extra context details
    if e_extra:
        line2 = f"*{e_desc}* : _{e_extra}_"
    else:
        line2 = f"*{e_desc}*"
    
    spacing = " " * 12
    line3 = f"📅 {e_date}{spacing}⏰ {e_time}"
    
    return f"{line1}\n{line2}\n{line3}"
        
def build_batched_msg(pair: str, price: Any, ts: int, items: List[Tuple[str, str]], score: Optional[float] = None, total: Optional[float] = None) -> str:
    price_str = _format_price(price)
    date_str = format_ist_time(ts, '%d-%m-%Y')
    time_str = format_ist_time(ts, '%H:%M IST')
    
    e_pair = escape_markdown_v2(pair)
    e_score = escape_markdown_v2(_fmt_score(score, total))
    e_price = escape_markdown_v2(price_str)
    e_date = escape_markdown_v2(date_str)
    e_time = escape_markdown_v2(time_str)
    spacing = " " * 12
    
    if not items:
        return f"*{e_pair}{e_score}* \\- *{e_price}*\n🗓️ {e_date}{spacing}🕙 {e_time}"
    
    headline_emoji = items[0][0].split(" ", 1)[0] if items[0][0] else "📊"
    e_headline_emoji = escape_markdown_v2(headline_emoji)
    
    line1 = f"{e_headline_emoji} *{e_pair}{e_score}* \\- *{e_price}*"
    
    condensed = len(items) > 2
    alert_lines = []
    for idx, (title, extra) in enumerate(items):
        parts = title.split(" ", 1)
        description = parts[1] if len(parts) == 2 else title
        e_desc = escape_markdown_v2(description)

        is_last = (idx == len(items) - 1)
        prefix = "➤" if is_last else "├➤"

        if condensed:
            alert_lines.append(f"{prefix} *{e_desc}*")
        else:
            extra_clean = _clean_extra_text(extra)
            e_extra = escape_markdown_v2(extra_clean)
            if e_extra:
                alert_lines.append(f"{prefix} *{e_desc}* : _{e_extra}_")
            else:
                alert_lines.append(f"{prefix} *{e_desc}*")
    
    body = "\n".join(alert_lines)
    datetime_line = f"📆  {e_date}{spacing}⏰ {e_time}"
    
    return f"{line1}\n{body}\n{datetime_line}"

def create_pivot_alert(level: str, is_buy: bool) -> Dict[str, Any]:
    """Factory function to create pivot alert definitions (check_fn/extra_fn are lambdas closing over `level`/`is_buy`)"""
    if is_buy:
        return {
            "key": f"pivot_up_{level}",
            "title": f"🟢⬆️ Cross above {level}",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: (
            ctx.get("buy_common_wick", False) and
            get_pivot_alert_info(ctx, level, is_buy=True)[0]
        ),
            "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: (
                f"${ctx['pivots'][level]:,.2f}"
                f"[Dist: {abs(ctx['pivots'][level] - ctx['close_curr'])/ctx['pivots'][level]*100:.2f}%]"
            ),
            "requires": ["pivots"]
        }
    else:
        return {
            "key": f"pivot_down_{level}",
            "title": f"🔴⬇️ Cross below {level}",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: (
            ctx.get("sell_common_wick", False) and
            get_pivot_alert_info(ctx, level, is_buy=False)[0]
        ),
            "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: (
                f"${ctx['pivots'][level]:,.2f}"
                f"[Dist: {abs(ctx['pivots'][level] - ctx['close_curr'])/ctx['pivots'][level]*100:.2f}%]"
            ),
            "requires": ["pivots"]
        }

@dataclass(frozen=True, slots=True)
class AlertRule:
    key: str
    title: str
    check_fn: Callable[[Any, Any, Any, Any], bool]
    extra_fn: Callable[[Any, Any, Any, Any, Dict[str, Any]], str]
    requires: List[str]

    def __post_init__(self) -> None:
        if not callable(self.check_fn):
            raise TypeError(f"Alert '{self.key}': check_fn must be callable")
        if not callable(self.extra_fn):
            raise TypeError(f"Alert '{self.key}': extra_fn must be callable")
        if not isinstance(self.requires, list):
            raise TypeError(f"Alert '{self.key}': requires must be a list")

_ALERT_DEFINITIONS_RAW: List[Dict[str, Any]] = [

    {"key":"ppo_signal_up","title":"🟢 PPO cross▲signal","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common_relaxed_wick",False) and (ppo.get("prev",np.nan)<=ppo_sig.get("prev",np.nan)) and (ppo.get("curr",np.nan)>ppo_sig.get("curr",np.nan)) and (ppo.get("curr",np.nan)<Constants.PPO_SIGNAL_CROSS_MAX_BUY or rsi.get("curr",np.nan)<Constants.RSI_SIGNAL_CROSS_MAX_BUY) and (ctx.get("ppo_gate_curr",np.nan)<Constants.PPO_RSI_GUARD_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} vs Sig {ppo_sig.get('curr',0):.2f} | RSI {rsi.get('curr',0):.1f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":["ppo","ppo_signal","ppo_alerts"]},
    {"key":"ppo_signal_down","title":"🔴 PPO cross▼signal","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common_relaxed_wick",False) and (ppo.get("prev",np.nan)>=ppo_sig.get("prev",np.nan)) and (ppo.get("curr",np.nan)<ppo_sig.get("curr",np.nan)) and (ppo.get("curr",np.nan)>Constants.PPO_SIGNAL_CROSS_MIN_SELL or rsi.get("curr",np.nan)>Constants.RSI_SIGNAL_CROSS_MIN_SELL) and (ctx.get("ppo_gate_curr",np.nan)>Constants.PPO_RSI_GUARD_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} vs Sig {ppo_sig.get('curr',0):.2f} | RSI {rsi.get('curr',0):.1f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":["ppo","ppo_signal","ppo_alerts"]},
    {"key":"rsi_ema5_up","title":"🟢 RSI▲EMA5","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common_relaxed_wick",False) and (rsi.get("prev",50)<=rsi.get("ema_prev",50)) and (rsi.get("curr",50)>rsi.get("ema_curr",50)) and (rsi.get("curr",50)<ctx.get("rsi_adaptive_buy",60)) and (ppo.get("curr",np.nan)<Constants.PPO_SIGNAL_CROSS_MAX_BUY or rsi.get("curr",50)<Constants.RSI_SIGNAL_CROSS_MAX_BUY) and (ctx.get("ppo_gate_curr",np.nan)<Constants.PPO_RSI_GUARD_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"RSI {rsi.get('curr',50):.2f} ▲EMA5 {rsi.get('ema_curr',50):.2f} | cap {ctx.get('rsi_adaptive_buy',0):.1f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":["rsi","rsi_alerts"]},
    {"key":"rsi_ema5_down","title":"🔴 RSI▼EMA5","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common_relaxed_wick",False) and (rsi.get("prev",50)>=rsi.get("ema_prev",50)) and (rsi.get("curr",50)<rsi.get("ema_curr",50)) and (rsi.get("curr",50)>ctx.get("rsi_adaptive_sell",40)) and (ppo.get("curr",np.nan)>Constants.PPO_SIGNAL_CROSS_MIN_SELL or rsi.get("curr",50)>Constants.RSI_SIGNAL_CROSS_MIN_SELL) and (ctx.get("ppo_gate_curr",np.nan)>Constants.PPO_RSI_GUARD_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"RSI {rsi.get('curr',50):.2f} ▼EMA5 {rsi.get('ema_curr',50):.2f} | cap {ctx.get('rsi_adaptive_sell',0):.1f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":["rsi","rsi_alerts"]},
    {"key":"vwap_up","title":"🔵▲ VWAP Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common_relaxed_wick",False) and (ppo.get("curr",np.nan)<Constants.PPO_SIGNAL_CROSS_MAX_BUY or rsi.get("curr",np.nan)<Constants.RSI_SIGNAL_CROSS_MAX_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"VWAP {ctx.get('vwap_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":["vwap"]},
    {"key":"vwap_down","title":"🟣▼ VWAP Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common_relaxed_wick",False) and (ppo.get("curr",np.nan)>Constants.PPO_SIGNAL_CROSS_MIN_SELL or rsi.get("curr",np.nan)>Constants.RSI_SIGNAL_CROSS_MIN_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"VWAP {ctx.get('vwap_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":["vwap"]},
    {"key":"cloud_cross_up","title":"☁️🟢 Cloud Up Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common_relaxed_wick",False) and (ppo.get("curr",np.nan)<Constants.PPO_SIGNAL_CROSS_MAX_BUY or rsi.get("curr",np.nan)<Constants.RSI_SIGNAL_CROSS_MAX_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Cloud Upper {ctx.get('cloud_upper_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"cloud_cross_down","title":"☁️🔴 Cloud Down Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common_relaxed_wick",False) and (ppo.get("curr",np.nan)>Constants.PPO_SIGNAL_CROSS_MIN_SELL or rsi.get("curr",np.nan)>Constants.RSI_SIGNAL_CROSS_MIN_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Cloud Lower {ctx.get('cloud_lower_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":[]}, 
    { "key": "ob_reversal_buy", "title": "🟢🏛️ Order Block Reversal BUY", "check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_trend_common_relaxed",False) and ctx.get("ob_gate_ok_buy",False) and ctx.get("wick_or_pattern_buy",False) and (ppo.get("curr",np.nan) <Constants.PPO_SIGNAL_CROSS_MAX_BUY or rsi.get("curr",np.nan) <Constants.RSI_SIGNAL_CROSS_MAX_BUY)), "extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"{ctx.get('ob_gate_reason') or 'Demand OB reversed'} | PPO {ppo.get('curr',0):.2f} RSI {rsi.get('curr',0):.1f}", "requires":[]},
    { "key": "ob_reversal_sell", "title": "🔴🏛 Order Block Reversal SELL", "check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_trend_common_relaxed",False) and ctx.get("ob_gate_ok_sell",False) and ctx.get("wick_or_pattern_sell",False) and (ppo.get("curr",np.nan) >Constants.PPO_SIGNAL_CROSS_MIN_SELL or rsi.get("curr",np.nan) >Constants.RSI_SIGNAL_CROSS_MIN_SELL)), "extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"{ctx.get('ob_gate_reason') or 'Supply OB reversed'} | PPO {ppo.get('curr',0):.2f} RSI {rsi.get('curr',0):.1f}", "requires":[]},
    {"key":"ppo_zero_up","title":"🟢 PPO cross▲0","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common_wick",False) and (ppo.get("prev",np.nan)<=0.0) and (ppo.get("curr",np.nan)>0.0) and (ctx.get("ppo_gate_curr",np.nan)<Constants.PPO_RSI_GUARD_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":["ppo","ppo_alerts"]},
    {"key":"ppo_zero_down","title":"🔴 PPO cross▼0","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common_wick",False) and (ppo.get("prev",np.nan)>=0.0) and (ppo.get("curr",np.nan)<0.0) and (ctx.get("ppo_gate_curr",np.nan)>Constants.PPO_RSI_GUARD_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":["ppo","ppo_alerts"]},
    {"key":"ppo_adaptive_up","title":"🟢 PPO cross▲adapt","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common_wick",False) and (ppo.get("prev",np.nan)<=ctx.get("ppo_adaptive_threshold",0.11)) and (ppo.get("curr",np.nan)>ctx.get("ppo_adaptive_threshold",0.11)) and (ctx.get("ppo_gate_curr",np.nan)<Constants.PPO_RSI_GUARD_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} vs adapt {ctx.get('ppo_adaptive_threshold',0):.3f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":["ppo","ppo_alerts"]},
    {"key":"ppo_adaptive_down","title":"🔴 PPO cross▼adapt","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common_wick",False) and (ppo.get("prev",np.nan)>=-ctx.get("ppo_adaptive_threshold",0.11)) and (ppo.get("curr",np.nan)<-ctx.get("ppo_adaptive_threshold",0.11)) and (ctx.get("ppo_gate_curr",np.nan)>Constants.PPO_RSI_GUARD_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} vs adapt {-ctx.get('ppo_adaptive_threshold',0):.3f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":["ppo","ppo_alerts"]},
    {"key":"rsi_cross_adaptive_up","title":"🟢 RSI▲adapt","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common_wick",False) and (rsi.get("curr",50)>rsi.get("ema_curr",50)) and (rsi.get("prev",50)<=ctx.get("rsi_adaptive_buy",60)) and (rsi.get("curr",50)>ctx.get("rsi_adaptive_buy",60)) and (ctx.get("ppo_gate_curr",np.nan)<Constants.PPO_RSI_GUARD_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"RSI {rsi.get('curr',50):.2f} ▲{ctx.get('rsi_adaptive_buy',0):.1f} | EMA5 {rsi.get('ema_curr',50):.2f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":["rsi","rsi_alerts"]},
    {"key":"rsi_cross_adaptive_down","title":"🔴 RSI▼adapt","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common_wick",False) and (rsi.get("curr",50)<rsi.get("ema_curr",50)) and (rsi.get("prev",50)>=ctx.get("rsi_adaptive_sell",40)) and (rsi.get("curr",50)<ctx.get("rsi_adaptive_sell",40)) and (ctx.get("ppo_gate_curr",np.nan)>Constants.PPO_RSI_GUARD_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"RSI {rsi.get('curr',50):.2f} ▼{ctx.get('rsi_adaptive_sell',0):.1f} | EMA5 {rsi.get('ema_curr',50):.2f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":["rsi","rsi_alerts"]},
    {"key":"hist_rma_buy","title":"🔵⬆️ RMA Rev BUY","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common_wick",False) and ctx.get("hist_reversal_buy",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Hist ({ctx.get('hist_curr',0):.4f}) | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"hist_rma_sell","title":"🟣 ⬇️ RMA Rev SELL","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common_wick",False) and ctx.get("hist_reversal_sell",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Hist ({ctx.get('hist_curr',0):.4f}) | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"ppohist_buy","title":"🟢🔥 PPO Rev BUY","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common_wick",False) and ctx.get("ppohist_reversal_buy",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPOHist ({ctx.get('ppohist_curr',0):.4f}) | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":["ppohist"]},
    {"key":"ppohist_sell","title":"🔴🔥 PPO Rev SELL","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common_wick",False) and ctx.get("ppohist_reversal_sell",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPOHist ({ctx.get('ppohist_curr',0):.4f}) | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":["ppohist"]}, 
    {"key":"tk_conversion_up","title":"🌐🟢 Tenkan Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("buy_common_wick",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Conv {ctx.get('tk_conversion_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"tk_conversion_down","title":"🌐🔴 Tenkan Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("sell_common_wick",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Conv {ctx.get('tk_conversion_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":[]}, 
    {"key":"kijun_cross_up","title":"⚓🟢 Kijun Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("buy_common_wick",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Base {ctx.get('tk_base_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"kijun_cross_down","title":"⚓🔴 Kijun Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("sell_common_wick",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Base {ctx.get('tk_base_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":[]}, 
    {"key":"strong_reversal_buy","title":"🟢🔄 Strong Reversal BUY","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("strong_reversal_buy",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"{ctx.get('reversal_pattern_name','Reversal candle')} confluence confirmed","requires":["strong_reversal"]},
    {"key":"strong_reversal_sell","title":"🔴🔄 Strong Reversal SELL","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("strong_reversal_sell",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"{ctx.get('reversal_pattern_name','Reversal candle')} confluence confirmed","requires":["strong_reversal"]},
    {"key":"choch_buy","title":"🟢🔀 CHoCH BUY","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("choch_buy",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"{ctx.get('choch_reason') or 'Bullish change of character'}","requires":["choch"]},
    {"key":"choch_sell","title":"🔴🔀 CHoCH SELL","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("choch_sell",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"{ctx.get('choch_reason') or 'Bearish change of character'}","requires":["choch"]},
    {"key":"fib_reversal_buy","title":"🟢🌀 Fib Pivot Reversal BUY","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("fib_reversal_buy",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"{ctx.get('fib_reversal_reason') or 'Fibonacci zone reversal'}","requires":["fib_reversal"]},
    {"key":"fib_reversal_sell","title":"🔴🌀 Fib Pivot Reversal SELL","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("fib_reversal_sell",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"{ctx.get('fib_reversal_reason') or 'Fibonacci zone reversal'}","requires":["fib_reversal"]},
    {"key":"dynamic_flow_cross_buy","title":"🌊🟢 Dynamic Flow Cross BUY","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("dynamic_flow_cross_buy",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"{ctx.get('dynamic_flow_cross_reason') or 'Dynamic Flow Ribbon flipped bullish'}","requires":["dynamic_flow_cross"]},
    {"key":"dynamic_flow_cross_sell","title":"🌊🔴 Dynamic Flow Cross SELL","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("dynamic_flow_cross_sell",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"{ctx.get('dynamic_flow_cross_reason') or 'Dynamic Flow Ribbon flipped bearish'}","requires":["dynamic_flow_cross"]},
]

def _validate_pivot_cross(ctx: Dict[str, Any], level: str, is_buy: bool) -> Tuple[bool, Optional[str]]:
    pivots = ctx.get("pivots")
    if not pivots or level not in pivots:
        return False, "No pivot data"

    level_value = pivots[level]
    if level_value <= 0:
        return False, "Invalid pivot value"

    close_curr = ctx.get("close_curr")
    close_prev = ctx.get("close_prev")

    if close_curr is None or close_prev is None or np.isnan(close_curr) or np.isnan(close_prev):
        return False, "Missing or invalid close data"

    # Precise cross verification
    if is_buy:
        crossed = close_prev <= level_value < close_curr
    else:
        crossed = close_prev >= level_value > close_curr

    if not crossed:
        return False, "No pivot cross"

    price_diff_pct = (abs(level_value - close_curr) / level_value) * 100
    max_distance = cfg.PIVOT_MAX_DISTANCE_PCT

    if price_diff_pct > max_distance:
        return False, (
            f"Pivot too far: price {close_curr:.2f} is {price_diff_pct:.2f}% "
            f"away from {level} pivot {level_value:.2f} (max {max_distance}%)"
        )

    return True, None

def _build_resets(pair_name: str, context: dict, conditional_states: dict) -> List[Tuple[str, str, None]]:
    """Generic cross-reset engine. Emits INACTIVE updates when a cross that was
    previously ACTIVE has now reversed."""
    resets: List[Tuple[str, str, None]] = []

    def _add(up_key: str, down_key: str,
             curr: float, prev: float,
             up_thr_curr: float, up_thr_prev: float,
             down_thr_curr: float, down_thr_prev: float) -> None:
        if prev > up_thr_prev and curr <= up_thr_curr:
            rk = ALERT_KEYS.get(up_key)
            if rk and conditional_states.get(rk, False):
                resets.append((f"{pair_name}:{rk}", "INACTIVE", None))
        if prev < down_thr_prev and curr >= down_thr_curr:
            rk = ALERT_KEYS.get(down_key)
            if rk and conditional_states.get(rk, False):
                resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── PPO ──
    ppo_c, ppo_p = context["ppo_curr"], context["ppo_prev"]
    ps_c,  ps_p  = context["ppo_sig_curr"], context["ppo_sig_prev"]
    thr = context["ppo_adaptive_threshold"]
    _add(AlertKey.PPO_SIGNAL_UP, AlertKey.PPO_SIGNAL_DOWN, ppo_c, ppo_p, ps_c, ps_p, ps_c, ps_p)
    _add(AlertKey.PPO_ZERO_UP,   AlertKey.PPO_ZERO_DOWN,   ppo_c, ppo_p, 0.0, 0.0, 0.0, 0.0)
    _add(AlertKey.PPO_ADAPTIVE_UP, AlertKey.PPO_ADAPTIVE_DOWN, ppo_c, ppo_p, thr, thr, -thr, -thr)

    #  RSI ──
    rsi_c, rsi_p = context["rsi_curr"], context["rsi_prev"]
    ema_c, ema_p = context["rsi_ema_curr"], context["rsi_ema_prev"]
    _add(AlertKey.RSI_EMA5_UP, AlertKey.RSI_EMA5_DOWN, rsi_c, rsi_p, ema_c, ema_p, ema_c, ema_p)
    buy_thr, sell_thr = context["rsi_adaptive_buy"], context["rsi_adaptive_sell"]
    _add(AlertKey.RSI_CROSS_ADAPTIVE_UP, AlertKey.RSI_CROSS_ADAPTIVE_DOWN,
         rsi_c, rsi_p, buy_thr, buy_thr, sell_thr, sell_thr)

    # ── VWAP ──
    if context.get("vwap_available"):
        _add(AlertKey.VWAP_UP, AlertKey.VWAP_DOWN, context["close_curr"], context["close_prev"],
             context["vwap_curr"], context["vwap_prev"], context["vwap_curr"], context["vwap_prev"])
    else:
        for k in (AlertKey.VWAP_UP, AlertKey.VWAP_DOWN):
            rk = ALERT_KEYS.get(k)
            if rk and conditional_states.get(rk, False):
                resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── Cloud crosses (slow + fast) ──
    for up_k, down_k, cu, cu_p, cl, cl_p in (
        (AlertKey.CLOUD_CROSS_UP, AlertKey.CLOUD_CROSS_DOWN,
         "cloud_upper_curr", "cloud_upper_prev", "cloud_lower_curr", "cloud_lower_prev"),
    ):
        cu_c, cu_pr = context.get(cu), context.get(cu_p)
        cl_c, cl_pr = context.get(cl), context.get(cl_p)
        if all(v is not None and not np.isnan(v) for v in (cu_c, cu_pr, cl_c, cl_pr)):
            _add(up_k, down_k, context["close_curr"], context["close_prev"], cu_c, cu_pr, cl_c, cl_pr)
        else:
            for k in (up_k, down_k):
                rk = ALERT_KEYS.get(k)
                if rk and conditional_states.get(rk, False):
                    resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── Conversion / Kijun / Fast Tenkan ──
    for up_k, down_k, conv, conv_p in (
        (AlertKey.TK_CONVERSION_UP, AlertKey.TK_CONVERSION_DOWN, "tk_conversion_curr", "tk_conversion_prev"),
        (AlertKey.KIJUN_CROSS_UP,   AlertKey.KIJUN_CROSS_DOWN,   "tk_base_curr",       "tk_base_prev"),
    ):
        c_c, c_p = context.get(conv), context.get(conv_p)
        if c_c is not None and c_p is not None and not np.isnan(c_c) and not np.isnan(c_p):
            _add(up_k, down_k, context["close_curr"], context["close_prev"], c_c, c_p, c_c, c_p)
        else:
            for k in (up_k, down_k):
                rk = ALERT_KEYS.get(k)
                if rk and conditional_states.get(rk, False):
                    resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── Hist RMA ──
    hist_c, hist_m1 = context["hist_curr"], context["hist_m1"]
    hist_eps = max(1e-10, abs(context["close_curr"]) * 1e-6)  # starting point — tune to your pairs
    for k, cond in ((AlertKey.HIST_RMA_BUY,  np.isnan(hist_c) or hist_c <= hist_eps or hist_c <= hist_m1),
                    (AlertKey.HIST_RMA_SELL, np.isnan(hist_c) or hist_c >= -hist_eps or hist_c >= hist_m1)):
        rk = ALERT_KEYS.get(k)
        if rk and conditional_states.get(rk, False) and cond:
            resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    ph_c, ph_m1 = context["ppohist_curr"], context["ppohist_m1"]
    for k, cond in ((AlertKey.PPOHIST_BUY,  np.isnan(ph_c) or ph_c <= 1e-8 or ph_c <= ph_m1),
                    (AlertKey.PPOHIST_SELL, np.isnan(ph_c) or ph_c >= -1e-8 or ph_c >= ph_m1)):
        rk = ALERT_KEYS.get(k)
        if rk and conditional_states.get(rk, False) and cond:
            resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── Order Block reversal ──
    for k, ok_key in ((AlertKey.OB_REVERSAL_BUY, "ob_gate_ok_buy"), (AlertKey.OB_REVERSAL_SELL, "ob_gate_ok_sell")):
        rk = ALERT_KEYS.get(k)
        if rk and conditional_states.get(rk, False) and not context.get(ok_key):
            resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── Strong reversal candle (engulfing/piercing/star/soldiers/tweezer/harami/marubozu/pinbar) ──
    for k, ok_key in ((AlertKey.STRONG_REVERSAL_BUY, "strong_reversal_buy"), (AlertKey.STRONG_REVERSAL_SELL, "strong_reversal_sell")):
        rk = ALERT_KEYS.get(k)
        if rk and conditional_states.get(rk, False) and not context.get(ok_key):
            resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── CHoCH liquidity-sweep reversal ──
    for k, ok_key in ((AlertKey.CHOCH_BUY, "choch_buy"), (AlertKey.CHOCH_SELL, "choch_sell")):
        rk = ALERT_KEYS.get(k)
        if rk and conditional_states.get(rk, False) and not context.get(ok_key):
            resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── Dynamic Flow Ribbon crossover ──
    for k, ok_key in ((AlertKey.DYNAMIC_FLOW_CROSS_BUY, "dynamic_flow_cross_buy"), (AlertKey.DYNAMIC_FLOW_CROSS_SELL, "dynamic_flow_cross_sell")):
        rk = ALERT_KEYS.get(k)
        if rk and conditional_states.get(rk, False) and not context.get(ok_key):
            resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── Fibonacci Pivot Reversal ──
    for k, ok_key in ((AlertKey.FIB_REVERSAL_BUY, "fib_reversal_buy"), (AlertKey.FIB_REVERSAL_SELL, "fib_reversal_sell")):
        rk = ALERT_KEYS.get(k)
        if rk and conditional_states.get(rk, False) and not context.get(ok_key):
            resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── Pivots ──
    piv = context.get("pivots", {})
    close_c, close_p = context["close_curr"], context["close_prev"]
    if not piv:
        for lvl in set(PIVOT_LEVELS_BUY + PIVOT_LEVELS_SELL):
            for prefix in ("pivot_up_", "pivot_down_"):
                k = f"{prefix}{lvl}"
                rk = ALERT_KEYS.get(k)
                if rk and conditional_states.get(rk, False):
                    resets.append((f"{pair_name}:{rk}", "INACTIVE", None))
    else:
        for lvl, val in piv.items():
            up_k = f"pivot_up_{lvl}"
            rk = ALERT_KEYS.get(up_k)
            if rk and conditional_states.get(rk, False) and close_p > val and close_c <= val:
                resets.append((f"{pair_name}:{rk}", "INACTIVE", None))
            down_k = f"pivot_down_{lvl}"
            rk = ALERT_KEYS.get(down_k)
            if rk and conditional_states.get(rk, False) and close_p < val and close_c >= val:
                resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    return resets

def get_pivot_alert_info(ctx: Dict[str, Any], level: str, is_buy: bool) -> Tuple[bool, Optional[str]]:
    cache_key = f"_pivot_cache_{level}_{'buy' if is_buy else 'sell'}"
    
    if cache_key not in ctx:
        ctx[cache_key] = _validate_pivot_cross(ctx, level, is_buy)
    
    return ctx[cache_key]

BUY_PIVOT_DEFS = [AlertRule(**create_pivot_alert(level, is_buy=True))
                  for level in PIVOT_LEVELS_BUY]

SELL_PIVOT_DEFS = [AlertRule(**create_pivot_alert(level, is_buy=False))
                   for level in PIVOT_LEVELS_SELL]

ALERT_DEFINITIONS: List[AlertRule] = [AlertRule(**d) for d in _ALERT_DEFINITIONS_RAW]
ALERT_DEFINITIONS.extend(BUY_PIVOT_DEFS)
ALERT_DEFINITIONS.extend(SELL_PIVOT_DEFS)

ALERT_DEFINITIONS_MAP = {d.key: d for d in ALERT_DEFINITIONS}

ALERT_KEYS: Dict[str, str] = {
    d.key: f"ALERT:{d.key.upper()}" for d in ALERT_DEFINITIONS
}

AlertKey = StrEnum("AlertKey", {k.upper(): k for k in ALERT_KEYS})

logger.debug("Alert keys initialized: %s mappings", len(ALERT_KEYS))

BUY_ALERT_KEYS: Set[str] = {
    "ppo_signal_up", "ppo_zero_up", "ppo_adaptive_up",
    "rsi_ema5_up", "rsi_cross_adaptive_up", "vwap_up", "hist_rma_buy", "ppohist_buy",
    "cloud_cross_up", "tk_conversion_up", "kijun_cross_up", "ob_reversal_buy", 
    "strong_reversal_buy", "choch_buy", "dynamic_flow_cross_buy", "fib_reversal_buy",
}
BUY_ALERT_KEYS.update(f"pivot_up_{level}" for level in PIVOT_LEVELS_BUY)

SELL_ALERT_KEYS: Set[str] = {
    "ppo_signal_down", "ppo_zero_down", "ppo_adaptive_down",
    "rsi_ema5_down", "rsi_cross_adaptive_down", "vwap_down", "hist_rma_sell", "ppohist_sell",
    "cloud_cross_down", "tk_conversion_down", "kijun_cross_down", "ob_reversal_sell",
    "strong_reversal_sell", "choch_sell", "dynamic_flow_cross_sell", "fib_reversal_sell",
}
SELL_ALERT_KEYS.update(f"pivot_down_{level}" for level in PIVOT_LEVELS_SELL)

def validate_alert_definitions() -> None:
    errors = []

    keys_seen = set()
    for def_ in ALERT_DEFINITIONS:
        key = def_.key
        if key in keys_seen:
            errors.append(f"Duplicate alert key: {key}")
        keys_seen.add(key)

    for def_ in ALERT_DEFINITIONS:
        if def_.key not in ALERT_KEYS:
            errors.append(f"Alert key {def_.key} missing from ALERT_KEYS mapping")
        if def_.key not in BUY_ALERT_KEYS and def_.key not in SELL_ALERT_KEYS:
            errors.append(f"Alert key {def_.key} missing from BUY_ALERT_KEYS/SELL_ALERT_KEYS")

    if errors:
        error_msg = "❌ ALERT DEFINITION VALIDATION FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.critical(error_msg)
        raise ValueError(error_msg)
    
    logger.debug(f"✅ Validated {len(ALERT_DEFINITIONS)} alert definitions ({len(ALERT_KEYS)} keys)")

validate_alert_definitions()

async def _eval_alerts(gr: GateResult, data_5m: PriceData, data_daily: Optional[Dict[str, np.ndarray]],
    reference_time: int, sdb: RedisStateStore, correlation_id: str, logger_pair: logging.Logger
) -> Union[Tuple[Dict[str, Any], Dict[str, bool], List[Tuple[str, str, str]]], Tuple[str, Dict[str, Any]], None]:
    pair_name = gr.pair_name
    i15 = gr.i15
    data_15m = gr.data_15m
    close_curr = gr.close_curr
    close_prev = gr.close_prev
    is_green, is_red = gr.is_green, gr.is_red
    is_valid_for_buy, is_valid_for_sell = gr.is_valid_for_buy, gr.is_valid_for_sell
    buy_wick_ratio, sell_wick_ratio = gr.buy_wick_ratio, gr.sell_wick_ratio
    rma50_15_val, rma200_5_val = gr.rma50_15_val, gr.rma200_5_val
    cloud_up, cloud_down = gr.cloud_up, gr.cloud_down
    cloud_upper_val, cloud_lower_val = gr.cloud_upper_val, gr.cloud_lower_val
    cloud_upper_prev, cloud_lower_prev = gr.cloud_upper_prev, gr.cloud_lower_prev
    ichimoku_gate_ok_buy, ichimoku_gate_ok_sell = gr.ichimoku_gate_ok_buy, gr.ichimoku_gate_ok_sell
    cloud_group_ok_buy, cloud_group_ok_sell = gr.cloud_group_ok_buy, gr.cloud_group_ok_sell
    tk_conversion_curr, tk_conversion_prev = gr.tk_conversion_curr, gr.tk_conversion_prev
    tk_base_curr, tk_base_prev = gr.tk_base_curr, gr.tk_base_prev
    tk_guard_ok_buy, tk_guard_ok_sell = gr.tk_guard_ok_buy, gr.tk_guard_ok_sell
    oscillator_group_ok_buy, oscillator_group_ok_sell = gr.oscillator_group_ok_buy, gr.oscillator_group_ok_sell
    ppo_gate_arr, ppo_gate_signal_arr = gr.ppo_gate_arr, gr.ppo_gate_signal_arr
    ppo_gate_curr, ppo_gate_prev = gr.ppo_gate_curr, gr.ppo_gate_prev
    ppo_gate_sig_curr, ppo_gate_sig_prev = gr.ppo_gate_sig_curr, gr.ppo_gate_sig_prev
    rsi_guard_smooth_curr, rsi_guard_ema_curr = gr.rsi_guard_smooth_curr, gr.rsi_guard_ema_curr
    rma_cloud_fast_curr = gr.rma_cloud_fast_curr
    rma_cloud_ok_buy, rma_cloud_ok_sell = gr.rma_cloud_ok_buy, gr.rma_cloud_ok_sell
    trend_gate_ok_buy, trend_gate_ok_sell = gr.trend_gate_ok_buy, gr.trend_gate_ok_sell
    adx_adaptive_threshold = gr.adx_adaptive_threshold
    momentum_count = gr.momentum_count
    effective_cpr_ok = gr.effective_cpr_ok
    cpr_adaptive_min_pct_move = gr.cpr_adaptive_min_pct_move
    move_from_prev_close_ok = gr.move_from_prev_close_ok
    ppo_adaptive_threshold = gr.ppo_adaptive_threshold
    rsi_adaptive_buy, rsi_adaptive_sell = gr.rsi_adaptive_buy, gr.rsi_adaptive_sell
    buy_common, sell_common = gr.buy_common, gr.sell_common
    buy_trend_common, sell_trend_common = gr.buy_trend_common, gr.sell_trend_common
    buy_trend_common_relaxed, sell_trend_common_relaxed = gr.buy_trend_common_relaxed, gr.sell_trend_common_relaxed
    close_prev_invalid = gr.close_prev_invalid
    ob_gate_ok_buy, ob_gate_ok_sell = gr.ob_gate_ok_buy, gr.ob_gate_ok_sell
    ob_gate_reason = gr.ob_gate_reason
    choch_gate_ok_buy, choch_gate_ok_sell = gr.choch_gate_ok_buy, gr.choch_gate_ok_sell
    choch_reason = gr.choch_reason
    choch_fvg_buy, choch_fvg_sell = gr.choch_fvg_buy, gr.choch_fvg_sell
    choch_poi_tap_buy, choch_poi_tap_sell = gr.choch_poi_tap_buy, gr.choch_poi_tap_sell
    dynamic_flow_cross_up, dynamic_flow_cross_down = gr.dynamic_flow_cross_up, gr.dynamic_flow_cross_down

    try:
        alert_indicators = await asyncio.to_thread(
            calculate_alert_indicators_numpy, data_15m.as_dict(), data_5m.as_dict(), data_daily, reference_time
        )
        if alert_indicators is None:
            logger_pair.error(f"Skipping {pair_name}: alert indicators failed")
            return None

        indicators = IndicatorCache.from_dicts(gr.gate_indicators, alert_indicators)

        critical_indicators = ["ppo", "ppo_signal", "smooth_rsi", "smooth_rsi_ema"]
        is_valid, msg = validate_indicators_dict(indicators.as_dict(), critical_indicators)
        if not is_valid:
            logger_pair.warning(f"Skipping {pair_name}: {msg}")
            return None

        ppo = indicators.ppo
        ppo_signal = indicators.ppo_signal
        smooth_rsi = indicators.smooth_rsi
        smooth_rsi_ema = indicators.smooth_rsi_ema
        vwap = indicators.vwap
        hist_rma = indicators.hist_rma
        piv = indicators.pivots or {}

        ppo_sig_curr = ppo_signal[i15]
        ppo_sig_prev = ppo_signal[i15 - 1] if i15 >= 1 else ppo_signal[i15]
        ppo_curr = ppo[i15]
        ppo_prev = ppo[i15 - 1] if i15 >= 1 else ppo[i15]
        ppohist_curr = ppo_gate_curr - ppo_gate_sig_curr
        rsi_curr = smooth_rsi[i15]
        rsi_prev = smooth_rsi[i15 - 1] if i15 >= 1 else smooth_rsi[i15]
        rsi_ema_curr = smooth_rsi_ema[i15]
        rsi_ema_prev = smooth_rsi_ema[i15 - 1] if i15 >= 1 else smooth_rsi_ema[i15]

        vwap_enabled = cfg.ENABLE_VWAP
        vwap_available = False
        vwap_curr = None
        vwap_prev = None
        if vwap_enabled and not close_prev_invalid and vwap is not None and len(vwap) > i15:
            try:
                vwap_curr = vwap[i15]
                vwap_prev = vwap[i15 - 1] if i15 >= 1 else vwap[i15]
                if (not np.isnan(vwap_curr) and not np.isnan(vwap_prev)
                        and vwap_curr > 0 and vwap_prev > 0):
                    vwap_available = True
                    if cfg.DEBUG_MODE:
                        logger_pair.debug(
                            f"[{pair_name}] VWAP OK: curr={vwap_curr:.4f}, prev={vwap_prev:.4f}"
                        )
                else:
                    if cfg.DEBUG_MODE:
                        logger_pair.debug(
                            f"[{pair_name}] VWAP invalid: curr={vwap_curr}, prev={vwap_prev}"
                        )
                    vwap_curr = None
                    vwap_prev = None
            except (IndexError, TypeError) as e:
                logger_pair.warning(f"[{pair_name}] VWAP access error: {e}")
                vwap_curr = None
                vwap_prev = None
        else:
            if vwap_enabled and cfg.DEBUG_MODE:
                logger_pair.debug(
                    f"[{pair_name}] VWAP unavailable: enabled={vwap_enabled}, "
                    f"vwap_is_none={vwap is None}, "
                    f"len={len(vwap) if vwap is not None else 0}, i15={i15}"
                )

# Reversal-candle pattern is detected once, unconditionally, so that
        reversal_bullish, reversal_bearish, reversal_pattern_name = detect_reversal_candle_pattern(data_15m, i15)

        # Shared "wick ratio OR strong reversal candle" condition, and the
        #   *_common_wick         = strict trend/volatility gate (buy_trend_common) + wick-or-pattern
        #   *_common_relaxed_wick = relaxed gate (buy_trend_common_relaxed, no volatility filter) + wick-or-pattern
        wick_or_pattern_buy = bool(buy_wick_ratio <= Constants.MIN_WICK_RATIO or reversal_bullish)
        wick_or_pattern_sell = bool(sell_wick_ratio <= Constants.MIN_WICK_RATIO or reversal_bearish)
        buy_common_wick = bool(buy_trend_common and wick_or_pattern_buy)
        sell_common_wick = bool(sell_trend_common and wick_or_pattern_sell)
        buy_common_relaxed_wick = bool(buy_trend_common_relaxed and wick_or_pattern_buy)
        sell_common_relaxed_wick = bool(sell_trend_common_relaxed and wick_or_pattern_sell)
   
        hist_curr = hist_rma[i15]
        hist_m1 = hist_rma[i15 - 1] if i15 >= 1 else 0.0
        hist_m2 = hist_rma[i15 - 2] if i15 >= 2 else 0.0
        hist_m3 = hist_rma[i15 - 3] if i15 >= 3 else 0.0

        ppohist_m1 = (ppo_gate_arr[i15-1] - ppo_gate_signal_arr[i15-1]) if i15 >= 1 else 0.0
        ppohist_m2 = (ppo_gate_arr[i15-2] - ppo_gate_signal_arr[i15-2]) if i15 >= 2 else 0.0
        ppohist_m3 = (ppo_gate_arr[i15-3] - ppo_gate_signal_arr[i15-3]) if i15 >= 3 else 0.0

        MIN_HIST_RMA_BARS_VALID = cfg.HIST_RMA_SLOW * 3
        has_valid_hist_rma = (
            cfg.ENABLE_HIST_RMA and
            i15 >= MIN_HIST_RMA_BARS_VALID and
            not np.isnan(hist_curr) and not np.isnan(hist_m1) and
            not np.isnan(hist_m2) and not np.isnan(hist_m3)
        )

        if not has_valid_hist_rma and cfg.DEBUG_MODE and cfg.ENABLE_HIST_RMA:
            skip_reason = (
                f"Hist RMA warmup" if i15 < MIN_HIST_RMA_BARS_VALID
                else f"Hist RMA NaN (idx={i15})"
            )
            logger_pair.debug(f"Skipping Hist RMA alerts: {skip_reason}")

        if not has_valid_hist_rma:
            hist_reversal_buy = False
            hist_reversal_sell = False
        else:
            hist_reversal_buy = (buy_common_wick and hist_curr > 0 and hist_m3 > hist_m2 > hist_m1 and hist_curr > hist_m1)
            hist_reversal_sell = (sell_common_wick and hist_curr < 0 and hist_m3 < hist_m2 < hist_m1 and hist_curr < hist_m1)

        min_ppohist_bars_valid = cfg.PPO_GATE_SLOW + cfg.PPO_GATE_SIGNAL + cfg.PPOHIST_WARMUP_BUFFER_BARS
        has_valid_ppohist = (
            cfg.ENABLE_PPO_GATE and
            i15 >= min_ppohist_bars_valid and
            not np.isnan(ppohist_curr) and not np.isnan(ppohist_m1) and
            not np.isnan(ppohist_m2) and not np.isnan(ppohist_m3)
        )
        if not has_valid_ppohist:
            ppohist_reversal_buy = False
            ppohist_reversal_sell = False
        else:
            ppohist_reversal_buy = (buy_common_wick and ppohist_curr > 0 and ppohist_m3 > ppohist_m2 > ppohist_m1 and ppohist_curr > ppohist_m1)
            ppohist_reversal_sell = (sell_common_wick and ppohist_curr < 0 and ppohist_m3 < ppohist_m2 < ppohist_m1 and ppohist_curr < ppohist_m1)

        if cfg.ENABLE_STRONG_REVERSAL_ALERT:
            strong_reversal_buy = (buy_trend_common_relaxed and reversal_bullish and (ppo_curr < Constants.PPO_SIGNAL_CROSS_MAX_BUY or rsi_curr < Constants.RSI_SIGNAL_CROSS_MAX_BUY))  
            strong_reversal_sell = (sell_trend_common_relaxed and reversal_bearish and (ppo_curr > Constants.PPO_SIGNAL_CROSS_MIN_SELL or rsi_curr > Constants.RSI_SIGNAL_CROSS_MIN_SELL))      
        else:
            strong_reversal_buy, strong_reversal_sell = False, False

        choch_reversal_bullish = False
        choch_reversal_bearish = False

        if cfg.ENABLE_CHOCH_ALERT:
            if cfg.ENABLE_STRONG_REVERSAL_ALERT:
                choch_reversal_bullish, choch_reversal_bearish = reversal_bullish, reversal_bearish
            elif choch_gate_ok_buy or choch_gate_ok_sell:
                choch_reversal_bullish, choch_reversal_bearish, _ = detect_reversal_candle_pattern(data_15m, i15)
            else:
                choch_reversal_bullish, choch_reversal_bearish = False, False

            choch_buy = bool(buy_trend_common_relaxed and choch_gate_ok_buy and wick_or_pattern_buy and (ppo_curr < Constants.PPO_SIGNAL_CROSS_MAX_BUY or rsi_curr < Constants.RSI_SIGNAL_CROSS_MAX_BUY))
            choch_sell = bool(sell_trend_common_relaxed and choch_gate_ok_sell and wick_or_pattern_sell and (ppo_curr > Constants.PPO_SIGNAL_CROSS_MIN_SELL or rsi_curr > Constants.RSI_SIGNAL_CROSS_MIN_SELL))
        else:
            choch_buy, choch_sell = False, False

        dynamic_flow_cross_buy, dynamic_flow_cross_sell = False, False
        dynamic_flow_cross_reason = None
        if cfg.ENABLE_DYNAMIC_FLOW_CROSS_ALERT:
            dynamic_flow_cross_buy = bool(buy_common_wick and dynamic_flow_cross_up)
            dynamic_flow_cross_sell = bool(sell_common_wick and dynamic_flow_cross_down)

            if dynamic_flow_cross_buy:
                dynamic_flow_cross_reason = "Price reclaimed Dynamic Flow line (bullish)"
            elif dynamic_flow_cross_sell:
                dynamic_flow_cross_reason = "Price lost Dynamic Flow line (bearish)"

        fib_reversal_buy, fib_reversal_sell = False, False
        fib_reversal_reason = None
        fib_reversal_votes_buy = fib_reversal_votes_sell = None

        if cfg.ENABLE_FIB_REVERSAL_ALERT:
            if buy_trend_common_relaxed and wick_or_pattern_buy and (ppo_curr < Constants.PPO_SIGNAL_CROSS_MAX_BUY or rsi_curr < Constants.RSI_SIGNAL_CROSS_MAX_BUY):
                vote_ok_buy, passed_buy, fib_reversal_votes_buy = _fib_reversal_confluence_vote(
                    data_15m.high, data_15m.low, data_15m.close, data_15m.volume, indicators.volume_ema,
                    smooth_rsi, ppo, i15, True, cfg, wick_or_pattern_buy,
                )
                fib_reversal_buy = bool(vote_ok_buy)
                if fib_reversal_buy:
                    passed_names = ", ".join(k for k, v in fib_reversal_votes_buy.items() if v)
                    fib_reversal_reason = f"Fib pivot reversal | {passed_buy}/4 confluence ({passed_names})"

            if sell_trend_common_relaxed and wick_or_pattern_sell and (ppo_curr > Constants.PPO_SIGNAL_CROSS_MIN_SELL or rsi_curr > Constants.RSI_SIGNAL_CROSS_MIN_SELL):
                vote_ok_sell, passed_sell, fib_reversal_votes_sell = _fib_reversal_confluence_vote(
                    data_15m.high, data_15m.low, data_15m.close, data_15m.volume, indicators.volume_ema,
                    smooth_rsi, ppo, i15, False, cfg, wick_or_pattern_sell,
                )
                fib_reversal_sell = bool(vote_ok_sell)
                if fib_reversal_sell:
                    passed_names = ", ".join(k for k, v in fib_reversal_votes_sell.items() if v)
                    fib_reversal_reason = f"Fib pivot reversal | {passed_sell}/4 confluence ({passed_names})"

        values_to_check = {
            'ppo_curr': ppo_curr, 'ppo_prev': ppo_prev,
            'rsi_curr': rsi_curr, 'rsi_prev': rsi_prev,
            'rsi_ema_curr': rsi_ema_curr, 'rsi_ema_prev': rsi_ema_prev,
            'ppo_sig_curr': ppo_sig_curr, 'ppo_sig_prev': ppo_sig_prev,
        }
        is_valid, msg = validate_indicator_values(values_to_check, list(values_to_check.keys()))
        if not is_valid:
            logger_pair.debug(msg)
            return None

        context = {
            "close_curr": close_curr, "close_prev": close_prev,
            "ppo_curr": ppo_curr, "ppo_prev": ppo_prev,
            "ppo_sig_curr": ppo_sig_curr, "ppo_sig_prev": ppo_sig_prev,
            "rsi_curr": rsi_curr, "rsi_prev": rsi_prev,
            "rsi_ema_curr": rsi_ema_curr, "rsi_ema_prev": rsi_ema_prev,
            "vwap_curr": vwap_curr, "vwap_prev": vwap_prev,
            "hist_curr": hist_curr, "hist_m1": hist_m1, "hist_m2": hist_m2, "hist_m3": hist_m3,
            "hist_reversal_buy": hist_reversal_buy, "hist_reversal_sell": hist_reversal_sell,
            "rma50_15_val": rma50_15_val, "rma200_5_val": rma200_5_val,
            "ppo_gate_curr": ppo_gate_curr, "ppo_gate_prev": ppo_gate_prev,
            "ppo_gate_sig_curr": ppo_gate_sig_curr, "ppo_gate_sig_prev": ppo_gate_sig_prev,
            "rsi_guard_smooth_curr": rsi_guard_smooth_curr, "rsi_guard_ema_curr": rsi_guard_ema_curr,
            "trend_gate_ok_buy": trend_gate_ok_buy, "trend_gate_ok_sell": trend_gate_ok_sell,
            "cloud_up": cloud_up, "cloud_down": cloud_down,
            "cloud_upper_curr": cloud_upper_val, "cloud_upper_prev": cloud_upper_prev,
            "cloud_lower_curr": cloud_lower_val, "cloud_lower_prev": cloud_lower_prev,
            "tk_guard_ok_buy": tk_guard_ok_buy, "tk_guard_ok_sell": tk_guard_ok_sell,
            "tk_conversion_curr": tk_conversion_curr, "tk_conversion_prev": tk_conversion_prev, "tk_base_curr": tk_base_curr, "tk_base_prev": tk_base_prev,
            "rma_cloud_ok_buy": rma_cloud_ok_buy, "rma_cloud_ok_sell": rma_cloud_ok_sell,
            "rma_cloud_fast_curr": rma_cloud_fast_curr, "rma_cloud_slow_curr": rma50_15_val,
            "ichimoku_gate_ok_buy": ichimoku_gate_ok_buy, "ichimoku_gate_ok_sell": ichimoku_gate_ok_sell, 
            "cloud_group_ok_buy": cloud_group_ok_buy, "cloud_group_ok_sell": cloud_group_ok_sell,
            "oscillator_group_ok_buy": oscillator_group_ok_buy, "oscillator_group_ok_sell": oscillator_group_ok_sell,
            "buy_common": buy_common, "sell_common": sell_common,
            "buy_trend_common": buy_trend_common, "sell_trend_common": sell_trend_common,
            "buy_trend_common_relaxed": buy_trend_common_relaxed, "sell_trend_common_relaxed": sell_trend_common_relaxed,
            "wick_or_pattern_buy": wick_or_pattern_buy, "wick_or_pattern_sell": wick_or_pattern_sell,
            "buy_common_wick": buy_common_wick, "sell_common_wick": sell_common_wick,
            "buy_common_relaxed_wick": buy_common_relaxed_wick, "sell_common_relaxed_wick": sell_common_relaxed_wick,
            "vwap_available": vwap_available,
            "vwap_enabled": cfg.ENABLE_VWAP and vwap_available,
            "ppohist_curr": ppohist_curr, "ppohist_m1": ppohist_m1,
            "ppohist_m2": ppohist_m2, "ppohist_m3": ppohist_m3,
            "ppohist_reversal_buy": ppohist_reversal_buy, "ppohist_reversal_sell": ppohist_reversal_sell,
            "adx_adaptive_threshold": adx_adaptive_threshold,
            "ppo_adaptive_threshold": ppo_adaptive_threshold,
            "rsi_adaptive_buy": rsi_adaptive_buy,
            "rsi_adaptive_sell": rsi_adaptive_sell,
            "buy_wick_ratio": buy_wick_ratio,
            "sell_wick_ratio": sell_wick_ratio,
            "is_green": is_green, "is_red": is_red,
            "pivots": piv if piv else {},
            "pivot_suppressions": [],
            "nr_cpr": indicators.nr_cpr,
            "cpr_ok": effective_cpr_ok,
            "momentum_count": momentum_count,
            "move_from_prev_close_ok": move_from_prev_close_ok, 
            "cpr_adaptive_min_pct_move": cpr_adaptive_min_pct_move,
            "ob_gate_ok_buy": ob_gate_ok_buy, "ob_gate_ok_sell": ob_gate_ok_sell,
            "ob_gate_reason": ob_gate_reason,
            "strong_reversal_buy": strong_reversal_buy, "strong_reversal_sell": strong_reversal_sell,
            "reversal_pattern_name": reversal_pattern_name,
            "reversal_bullish": reversal_bullish, "reversal_bearish": reversal_bearish,
            "choch_gate_ok_buy": choch_gate_ok_buy, "choch_gate_ok_sell": choch_gate_ok_sell,
            "choch_reason": choch_reason,
            "choch_fvg_buy": choch_fvg_buy, "choch_fvg_sell": choch_fvg_sell,
            "choch_buy": choch_buy, "choch_sell": choch_sell, 
            "choch_poi_tap_buy": choch_poi_tap_buy, "choch_poi_tap_sell": choch_poi_tap_sell,
            "fib_reversal_buy": fib_reversal_buy, "fib_reversal_sell": fib_reversal_sell,
            "fib_reversal_reason": fib_reversal_reason,
            "dynamic_flow_cross_buy": dynamic_flow_cross_buy, "dynamic_flow_cross_sell": dynamic_flow_cross_sell,
            "dynamic_flow_cross_reason": dynamic_flow_cross_reason,
        }
        ppo_ctx = {"curr": ppo_curr, "prev": ppo_prev}
        ppo_sig_ctx = {"curr": ppo_sig_curr, "prev": ppo_sig_prev}
        rsi_ctx = {"curr": rsi_curr, "prev": rsi_prev, "ema_curr": rsi_ema_curr, "ema_prev": rsi_ema_prev}

        alert_keys_to_check = []
        for d in ALERT_DEFINITIONS:
            key = d.key
            requires = d.requires
           
            skip = False
            if "pivots" in requires and (not cfg.ENABLE_PIVOT or not piv or not any(piv.values())):
                skip = True
            elif "strong_reversal" in requires and not cfg.ENABLE_STRONG_REVERSAL_ALERT:
                skip = True
            elif "choch" in requires and not cfg.ENABLE_CHOCH_ALERT:
                skip = True
            elif "fib_reversal" in requires and not cfg.ENABLE_FIB_REVERSAL_ALERT:
                skip = True
            elif "vwap" in requires and not vwap_available:
                skip = True
            elif "ppo_alerts" in requires and not cfg.ENABLE_PPO_ALERTS:
                skip = True
            elif "rsi_alerts" in requires and not cfg.ENABLE_RSI_ALERTS:
                skip = True
            elif "ppohist" in requires and not cfg.ENABLE_PPOHIST_ALERT:
                skip = True
            elif "ppo" in requires and ppo_ctx is None:
                skip = True
            elif "ppo_signal" in requires and ppo_sig_ctx is None:
                skip = True
            elif "rsi" in requires and rsi_ctx is None:
                skip = True
            
            if not skip:
                alert_keys_to_check.append(key)

        all_redis_alert_keys = list(ALERT_KEYS.values())
        previous_states = await sdb.batch_get_all_alert_states(pair_name, all_redis_alert_keys)
        disabled_alert_keys = await sdb.get_disabled_alert_keys()
        raw_alerts: List[Tuple[str, str, str]] = []

        # ── Registry for cross-based alerts (same pattern as _build_resets) ──
        _CROSS_HANDLERS = {
            "vwap": {
                "keys": {"vwap_up", "vwap_down"},
                "enabled": vwap_available,
                "validator": validate_vwap_cross,
                "ctx_args": ("close_prev", "close_curr", "vwap_prev", "vwap_curr"),
            },
            "cloud_cross": {
                "keys": {"cloud_cross_up", "cloud_cross_down"},
                "enabled": cfg.ENABLE_CLOUD_CROSS_ALERT,
                "validator": validate_cloud_cross,
                "ctx_args": ("close_prev", "close_curr", "cloud_upper_prev", "cloud_upper_curr",
                             "cloud_lower_prev", "cloud_lower_curr"),
            },
            "tk_conversion": {
                "keys": {"tk_conversion_up", "tk_conversion_down"},
                "enabled": cfg.ENABLE_TK_CONVERSION_CROSS,
                "validator": validate_conversion_cross,
                "ctx_args": ("close_prev", "close_curr", "tk_conversion_prev", "tk_conversion_curr"),
            },
            "kijun_cross": {
                "keys": {"kijun_cross_up", "kijun_cross_down"},
                "enabled": cfg.ENABLE_KIJUN_CROSS,
                "validator": validate_conversion_cross,
                "ctx_args": ("close_prev", "close_curr", "tk_base_prev", "tk_base_curr"),
            },
        }

        for alert_key in alert_keys_to_check:
            def_ = ALERT_DEFINITIONS_MAP.get(alert_key)
            if not def_:
                continue

            if alert_key in disabled_alert_keys: 
                if cfg.DEBUG_MODE:
                    logger_pair.debug(f"Skipping {alert_key}: brain-disabled (underperforming, per-key)")
                continue

            if alert_key in BUY_ALERT_KEYS:
                if not is_green:
                    logger_pair.debug(
                        f"[{pair_name}] 🚫 BLOCKED BUY: {alert_key} on RED candle! "
                        f"O={gr.open_curr:.2f} C={close_curr:.2f}"
                    )
                    continue

                choch_exception = (alert_key == "choch_buy" and choch_reversal_bullish)
                dynamic_flow_cross_exception = (alert_key == "dynamic_flow_cross_buy" and dynamic_flow_cross_buy)
                fib_reversal_exception = (alert_key == "fib_reversal_buy" and fib_reversal_buy)
                if not (is_valid_for_buy or reversal_bullish or choch_exception or dynamic_flow_cross_exception or fib_reversal_exception):
                    if cfg.DEBUG_MODE:
                        logger_pair.debug(f"Skipping {alert_key}: not valid for buy (wick/body fail, no reversal pattern)")
                    continue

            if alert_key in SELL_ALERT_KEYS:
                if not is_red:
                    logger_pair.debug(
                        f"[{pair_name}] 🚫 BLOCKED SELL: {alert_key} on GREEN candle! "
                        f"O={gr.open_curr:.2f} C={close_curr:.2f}"
                    )
                    continue

                choch_exception = (alert_key == "choch_sell" and choch_reversal_bearish)
                dynamic_flow_cross_exception = (alert_key == "dynamic_flow_cross_sell" and dynamic_flow_cross_sell)
                fib_reversal_exception = (alert_key == "fib_reversal_sell" and fib_reversal_sell)
                if not (is_valid_for_sell or reversal_bearish or choch_exception or dynamic_flow_cross_exception or fib_reversal_exception):
                    if cfg.DEBUG_MODE:
                        logger_pair.debug(f"Skipping {alert_key}: not valid for sell (wick/body fail, no reversal pattern)")
                    continue

            if is_green and alert_key.startswith("pivot_down"):
                logger_pair.debug(
                    f"[{pair_name}] LOGIC ERROR: GREEN candle firing pivot_down '{alert_key}'. "
                    f"Skipping to prevent false alert."
                )
                continue

            if is_red and alert_key.startswith("pivot_up"):
                logger_pair.debug(
                    f"[{pair_name}] LOGIC ERROR: RED candle firing pivot_up '{alert_key}'. "
                    f"Skipping to prevent false alert."
                )
                continue

            key = ALERT_KEYS[alert_key]
            trigger = False

            # ── Cross-alert dispatch ──
            handled = False
            for handler in _CROSS_HANDLERS.values():
                if alert_key not in handler["keys"]:
                    continue
                if not handler["enabled"]:
                    if cfg.DEBUG_MODE:
                        logger_pair.debug(f"Skipping {alert_key}: cross prerequisite disabled")
                    handled = True
                    break
                try:
                    is_buy_side = alert_key.endswith("_up")
                    args = [context[k] for k in handler["ctx_args"]] + [is_buy_side]
                    valid_cross, cross_reason = handler["validator"](*args)
                    if valid_cross:
                        trigger = def_.check_fn(context, ppo_ctx, ppo_sig_ctx, rsi_ctx)
                    elif cfg.DEBUG_MODE:
                        logger_pair.debug(f"{alert_key} cross check: {cross_reason}")
                except Exception as e:
                    logger_pair.debug(f"{alert_key} cross check failed: {e}", exc_info=True)
                handled = True
                break

            if not handled:
                if alert_key.startswith("pivot_up_") or alert_key.startswith("pivot_down_"):
                    level = alert_key.split("_")[-1]
                    is_buy = alert_key.startswith("pivot_up_")
                    try:
                        valid_cross, reason = get_pivot_alert_info(context, level, is_buy)
                        if not valid_cross and reason and piv:
                             context["pivot_suppressions"].append(f"{alert_key}: {reason}")
                        trigger = def_.check_fn(context, ppo_ctx, ppo_sig_ctx, rsi_ctx)
                    except Exception as e:
                        logger_pair.debug(f"Pivot alert check failed for {alert_key}: {e}", exc_info=True)
                        trigger = False
                else:
                    try:
                        trigger = def_.check_fn(context, ppo_ctx, ppo_sig_ctx, rsi_ctx)
                    except Exception as e:
                        logger_pair.debug(f"Alert check failed for {alert_key}: {e}", exc_info=True)
                        trigger = False

            if trigger and not previous_states.get(key, False):
                extra = ""
                try:
                    base_extra = def_.extra_fn(context, ppo_ctx, ppo_sig_ctx, rsi_ctx, None) or ""
                    extra = base_extra
                except Exception as e:
                    logger_pair.debug(f"Alert extra_fn failed for {alert_key}: {e}", exc_info=cfg.DEBUG_MODE)
                    extra = f"(Error: {str(e)[:100]})"
                raw_alerts.append((def_.title, extra, def_.key))
            
                if cfg.DEBUG_MODE:
                    logger_pair.debug(
                        f"✅ Alert FIRED: {alert_key} | "
                        f"buy_common={buy_common} sell_common={sell_common} | "
                        f"Candle: O={gr.open_curr:.2f} C={close_curr:.2f}"
                    )

        conditional_states = previous_states

        return context, conditional_states, raw_alerts

    except asyncio.CancelledError:
        logger_pair.warning(f"Evaluation cancelled for {pair_name}")
        raise
    except RuntimeError as e:
        logger_pair.critical(f"🚨 INVARIANT VIOLATION in {pair_name}: {e}")
        return pair_name, {
            "state": "INVARIANT_VIOLATION",
            "ts": int(time.time()),
            "summary": {
                "alerts": 0,
                "future_cloud": "neutral",
                "hist_rma": 0.0,
                "error": str(e)
            }
        }
    except Exception as e:
        logger_pair.exception(
            f"❌ Error in _eval_alerts for {pair_name}: {e} | Correlation: {correlation_id}"
        )
        return None

def _pct_returns(closes: np.ndarray) -> np.ndarray:
    """Simple bar-over-bar percent returns. Empty/too-short input -> empty array."""
    if closes is None or len(closes) < 2:
        return np.array([])
    prior = closes[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(closes) / prior
    return returns[np.isfinite(returns)]

def rolling_correlation(alt_returns: np.ndarray, btc_returns: np.ndarray, window: int = 20) -> float:
    """Pearson correlation of the last `window` 15m bar-returns. NaN if either
    series is shorter than `window` or has ~zero variance (corrcoef undefined)."""
    if len(alt_returns) < window or len(btc_returns) < window:
        return float("nan")
    a, b = alt_returns[-window:], btc_returns[-window:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

async def _apply_and_dispatch_alerts(gr: GateResult, context: Dict[str, Any], conditional_states: Dict[str, bool],
    raw_alerts: List[Tuple[str, str, str]], sdb: RedisStateStore, telegram_queue: TelegramQueue,
    fetcher: DataFetcher, symbol: str, correlation_id: str, logger_pair: logging.Logger,
    alerts_sent_ref: List[int], alerts_sent_lock: asyncio.Lock, max_alerts_per_run: int,
    data_5m: PriceData,
    confluence_score_buy: Optional[float] = None, confluence_total_buy: Optional[float] = None,
    confluence_votes_buy: Optional[Dict[str, bool]] = None,
    confluence_score_sell: Optional[float] = None, confluence_total_sell: Optional[float] = None,
    confluence_votes_sell: Optional[Dict[str, bool]] = None,
    macro_context: Optional[BtcMacroContext] = None,
    cluster_context: Optional[ClusterContext] = None) -> Tuple[str, Dict[str, Any]]:

    def _confluence_for(alert_key: str) -> Tuple[Optional[float], Optional[float], Optional[Dict[str, bool]]]:
        if alert_key in BUY_ALERT_KEYS:
            return confluence_score_buy, confluence_total_buy, confluence_votes_buy
        return confluence_score_sell, confluence_total_sell, confluence_votes_sell
    pair_name = gr.pair_name
    i15, ts_curr, reference_time = gr.i15, gr.ts_curr, gr.reference_time
    data_15m = gr.data_15m
    candle_info = gr.candle_info
    o, h, l, c = gr.o, gr.h, gr.l, gr.c
    close_curr, close_prev = gr.close_curr, gr.close_prev
    is_green, is_red = gr.is_green, gr.is_red
    is_valid_for_buy, is_valid_for_sell = gr.is_valid_for_buy, gr.is_valid_for_sell
    base_buy_trend, base_sell_trend = gr.base_buy_trend, gr.base_sell_trend
    rma50_15_val = gr.rma50_15_val
    cloud_up, cloud_down = gr.cloud_up, gr.cloud_down
    ichimoku_gate_ok_buy, ichimoku_gate_ok_sell = gr.ichimoku_gate_ok_buy, gr.ichimoku_gate_ok_sell
    confirmation_buy, confirmation_sell = gr.confirmation_buy, gr.confirmation_sell
    cloud_group_ok_buy, cloud_group_ok_sell = gr.cloud_group_ok_buy, gr.cloud_group_ok_sell
    tk_conversion_curr, tk_conversion_prev = gr.tk_conversion_curr, gr.tk_conversion_prev
    tk_base_curr, tk_base_prev = gr.tk_base_curr, gr.tk_base_prev
    oscillator_group_ok_buy, oscillator_group_ok_sell = gr.oscillator_group_ok_buy, gr.oscillator_group_ok_sell
    ppo_gate_curr, ppo_gate_sig_curr = gr.ppo_gate_curr, gr.ppo_gate_sig_curr
    ppo_gate_ok_buy, ppo_gate_ok_sell = gr.ppo_gate_ok_buy, gr.ppo_gate_ok_sell
    rsi_guard_smooth_curr, rsi_guard_ema_curr = gr.rsi_guard_smooth_curr, gr.rsi_guard_ema_curr
    rsi_guard_ok_buy, rsi_guard_ok_sell = gr.rsi_guard_ok_buy, gr.rsi_guard_ok_sell
    rma_cloud_fast_curr = gr.rma_cloud_fast_curr
    rma_cloud_ok_buy, rma_cloud_ok_sell = gr.rma_cloud_ok_buy, gr.rma_cloud_ok_sell
    adx_val, adx_adaptive_threshold, adx_ok = gr.adx_val, gr.adx_adaptive_threshold, gr.adx_ok
    rvol_ok = gr.rvol_ok
    ppo_adaptive_threshold = gr.ppo_adaptive_threshold
    rsi_adaptive_buy, rsi_adaptive_sell = gr.rsi_adaptive_buy, gr.rsi_adaptive_sell
    buy_common, sell_common = gr.buy_common, gr.sell_common

    hist_curr, hist_m1, hist_m2, hist_m3 = context["hist_curr"], context["hist_m1"], context["hist_m2"], context["hist_m3"]
    hist_reversal_buy, hist_reversal_sell = context["hist_reversal_buy"], context["hist_reversal_sell"]
    vwap_curr, vwap_prev, vwap_available = context["vwap_curr"], context["vwap_prev"], context["vwap_available"]
    ppo_curr, ppo_prev = context["ppo_curr"], context["ppo_prev"]
    rsi_curr, rsi_prev = context["rsi_curr"], context["rsi_prev"]
    rsi_ema_curr, rsi_ema_prev = context["rsi_ema_curr"], context["rsi_ema_prev"]

    all_state_changes = []
    try:
        resets_to_apply = _build_resets(pair_name, context, conditional_states)

        all_state_changes.extend(resets_to_apply)

        pivot_count = sum(1 for _, _, k in raw_alerts if k.startswith("pivot_"))
        if pivot_count > 3:
            logger_pair.warning(
                f"Limiting pivot alerts for {pair_name}: {pivot_count} triggered, keeping 3"
            )
            pivot_alerts = [(t, e, k) for t, e, k in raw_alerts if k.startswith("pivot_")][:3]
            other_alerts = [(t, e, k) for t, e, k in raw_alerts if not k.startswith("pivot_")]
            capped_alerts = other_alerts + pivot_alerts
        else:
            capped_alerts = raw_alerts

        alerts_to_send = capped_alerts[:cfg.MAX_ALERTS_PER_PAIR]

        cached_snapshot: Optional[CandleSnapshot] = None
        if alerts_to_send:
            has_reversal_alert = any(
                k in ("strong_reversal_buy", "strong_reversal_sell") for _, _, k in alerts_to_send
            )
            cached_snapshot = CandleSnapshot(
                timestamp=ts_curr, open=o, high=h, low=l, close=c,
                volume=candle_info["volume"],
                is_green=is_green, is_red=is_red,
                is_valid_for_buy=is_valid_for_buy, is_valid_for_sell=is_valid_for_sell,
                reversal_pattern_name=context.get("reversal_pattern_name", "") if has_reversal_alert else "",
                reversal_bullish=context.get("reversal_bullish", False) if has_reversal_alert else False,
                reversal_bearish=context.get("reversal_bearish", False) if has_reversal_alert else False,
            )
            reverified = independent_candle_reverify(
                data_15m=data_15m.as_dict(), candle_index=i15,
                cached=cached_snapshot,
                min_wick_ratio=Constants.MIN_WICK_RATIO,
                pair_name=pair_name, logger_pair=logger_pair,
            )
            if not reverified:
                logger_pair.warning(
                    f"[{pair_name}] Independent re-verify failed — alert suppressed. No dedup/coalesce "
                    f"claim was taken yet, so this will be re-attempted next run if the trigger persists."
                )
                alerts_to_send = []

            if alerts_to_send:
                cross_check = cross_check_15m_against_5m(
                    data_5m, ts_curr, cached_snapshot, pair_name, logger_pair
                )
                if cross_check is False:
                    alerts_to_send = []

        is_buy_batch = any(ak in BUY_ALERT_KEYS for _, _, ak in alerts_to_send) if alerts_to_send else False
        confluence_score = confluence_score_buy if is_buy_batch else confluence_score_sell
        confluence_total = confluence_total_buy if is_buy_batch else confluence_total_sell

        # ── Macro Context Modifier (LIVE when MACRO_CONTEXT_LIVE=True) ─────
        macro_shadow: Optional[Dict[str, Any]] = None
        macro_multiplier = 1.0
        if alerts_to_send and getattr(cfg, "ENABLE_MACRO_CONTEXT_GATE", False) and macro_context is not None:
            btc_bearish = macro_context.confirmation_sell and macro_context.adx_ok
            btc_bullish = macro_context.confirmation_buy and macro_context.adx_ok
            if (is_buy_batch and btc_bearish) or (not is_buy_batch and btc_bullish):
                corr = rolling_correlation(
                    _pct_returns(data_15m.close), _pct_returns(macro_context.closes),
                    window=Constants.MACRO_CORR_WINDOW,
                )
                if not np.isnan(corr):
                    if corr < Constants.MACRO_CORR_LOW:
                        macro_multiplier = 1.0
                    elif corr < Constants.MACRO_CORR_HIGH:
                        macro_multiplier = Constants.MACRO_MULT_MODERATE
                    else:
                        macro_multiplier = Constants.MACRO_MULT_FULL

                relative_strength = None
                if o not in (None, 0) and macro_context.open not in (None, 0):
                    alt_return_15m = (c - o) / o
                    btc_return_15m = (macro_context.close - macro_context.open) / macro_context.open
                    relative_strength = alt_return_15m - btc_return_15m
                    if relative_strength > 0:
                        macro_multiplier = max(1.0, macro_multiplier * Constants.MACRO_RS_EASE_FACTOR)

                macro_shadow = {
                    "correlation": None if np.isnan(corr) else round(corr, 3),
                    "relative_strength": None if relative_strength is None else round(relative_strength, 5),
                    "multiplier": round(macro_multiplier, 3),
                    "btc_bullish": btc_bullish, "btc_bearish": btc_bearish,
                }
                logger_pair.info(
                    f"[{pair_name}] 🌐 Macro context: corr={macro_shadow['correlation']}, "
                    f"RS={macro_shadow['relative_strength']}, multiplier={macro_multiplier:.2f}"
                    f"{' [LIVE]' if getattr(cfg, 'MACRO_CONTEXT_LIVE', False) else ' (SHADOW ONLY)'}"
                )

        # ── Existing confluence gate, but with macro multiplier ─────────────
        if alerts_to_send and cfg.ENABLE_CONFLUENCE_GATE and confluence_score is not None and confluence_total is not None:

        # ── Confluence gate threshold (shared by both checks below) ────────
            abs_floor = cfg.CONFLUENCE_MIN_ABS_SCORE
            if alerts_to_send and getattr(cfg, "ENABLE_PAIR_THRESHOLDS", False):
                pair_floor = await sdb.get_pair_threshold(pair_name)
                if pair_floor is not None:
                    abs_floor = pair_floor

            def _required_confluence(total: float) -> float:
                pct_floor = total * (cfg.CONFLUENCE_MIN_PCT / 100.0)
                req = max(pct_floor, abs_floor)
                if getattr(cfg, "MACRO_CONTEXT_LIVE", False):
                    req = req * macro_multiplier
                return req

            # ── Existing confluence gate, but with macro multiplier ─────────────
            if alerts_to_send and cfg.ENABLE_CONFLUENCE_GATE and confluence_score is not None and confluence_total is not None:
                required = _required_confluence(confluence_total)
                if macro_shadow is not None:
                    macro_shadow["would_block"] = confluence_score < required
                if confluence_score < required:
                    logger_pair.info(
                        f"[{pair_name}] Confluence gate blocked: {confluence_score:.1f}/{confluence_total:.1f} "
                        f"weighted score (need {required:.1f}, abs_floor={abs_floor:.1f}, "
                        f"macro_mult={macro_multiplier if getattr(cfg, 'MACRO_CONTEXT_LIVE', False) else 1.0:.2f})"
                    )
                    alerts_to_send = []

            # ── Correlation Cluster Penalty ("Beta Trap" filter) — LIVE ────────
            if (alerts_to_send and getattr(cfg, "ENABLE_CLUSTER_GATE", False)
                    and cluster_context is not None and confluence_score is not None):
                cluster_pct = cluster_context.buy_pct if is_buy_batch else cluster_context.sell_pct
                if cluster_pct > cfg.CLUSTER_PCT_THRESHOLD:
                    raw_score = confluence_score
                    confluence_score = confluence_score * (1 - cfg.CLUSTER_PENALTY_PCT)
                    logger_pair.info(
                        f"[{pair_name}] 🌐 Correlation cluster penalty: "
                        f"{'buy' if is_buy_batch else 'sell'} cluster at {cluster_pct:.0%} of "
                        f"{cluster_context.total_pairs} pairs (> {cfg.CLUSTER_PCT_THRESHOLD:.0%} threshold) — "
                        f"confluence score {raw_score:.1f} -> {confluence_score:.1f}"
                    )

            if alerts_to_send and cfg.ENABLE_CONFLUENCE_GATE and confluence_score is not None and confluence_total is not None:
                required = _required_confluence(confluence_total)
                if macro_shadow is not None:
                    macro_shadow["would_block"] = confluence_score < required
                if confluence_score < required:
                    logger_pair.info(
                        f"[{pair_name}] Confluence gate blocked dispatch: {confluence_score:.1f}/{confluence_total:.1f} weighted score (need {required:.1f}, abs_floor={abs_floor:.1f}, macro_mult={macro_multiplier if getattr(cfg, 'MACRO_CONTEXT_LIVE', False) else 1.0:.2f})"
                    )
                    alerts_to_send = []

        if alerts_to_send and getattr(cfg, "BRAIN_OOD_ENABLED", True):
            ood_survivors = []
            for alert_title, alert_extra, alert_key in alerts_to_send:
                _, _, alert_votes = _confluence_for(alert_key)
                if alert_votes is None:
                    ood_survivors.append((alert_title, alert_extra, alert_key))
                    continue
                current_count = sum(1 for v in alert_votes.values() if v)
                hist = await sdb.get_vote_count_history(alert_key)
                is_ood, detail = engine.is_vote_count_ood(
                    current_count, 
                    hist,
                    min_history=cfg.OOD_MIN_HISTORY,
                    margin=cfg.OOD_MARGIN,
                    p5=cfg.OOD_P5,
                    p95=cfg.OOD_P95,
                    relaxed_mode=cfg.OOD_RELAXED_MODE,
                )
                if is_ood:
                    logger_pair.warning(
                        f"[{pair_name}] ⚠️ Unusual vote pattern detected — trade blocked as precaution: "
                        f"{alert_key} vote count {detail['current_count']} outside historical range "
                        f"[{detail['hist_p5']:.1f}-{detail['hist_p95']:.1f}] (n={detail['n_history']})"
                    )
                    continue
                ood_survivors.append((alert_title, alert_extra, alert_key))
            alerts_to_send = ood_survivors

        if alerts_to_send and cfg.ENABLE_WIN_RATE_FILTER:
            alert_keys_to_check = [ak for _, _, ak in alerts_to_send]
            win_rate_map = await sdb.batch_get_alert_win_rates(
                pair_name, alert_keys_to_check, timeout=3.0
            )
            current_session = _get_session_from_ts(ts_curr) if getattr(cfg, "ENABLE_SESSION_FILTER", False) else None
            surviving_alerts = []
            brain_engine = None
            if cfg.ENABLE_BRAIN:
                try:
                    from brain_enhanced import BrainEngineV2 as BrainEngine
                    brain_engine = BrainEngine(sdb)
                except Exception as e:
                    logger_pair.debug(f"Brain engine init failed: {e}")

            for alert_title, alert_extra, alert_key in alerts_to_send:
                direction = "buy" if alert_key in BUY_ALERT_KEYS else "sell"
                win_rate, sample = win_rate_map.get(alert_key, (None, 0))
                failing_rate, fail_note = None, None
                if win_rate is not None and win_rate < cfg.MIN_WIN_RATE:
                    failing_rate = win_rate
                    fail_note = f"{win_rate:.0%} over {sample} samples (need >= {cfg.MIN_WIN_RATE:.0%})"
                if current_session is not None:
                    session_win_rate, session_sample = await sdb.get_alert_win_rate_session(
                        pair_name, alert_key, current_session
                    )
                    if session_win_rate is not None and session_win_rate < cfg.MIN_WIN_RATE:
                        if failing_rate is None or session_win_rate < failing_rate:
                            failing_rate = session_win_rate
                            fail_note = (
                                f"{session_win_rate:.0%} in {current_session} session over "
                                f"{session_sample} samples (need >= {cfg.MIN_WIN_RATE:.0%})"
                            )

                if failing_rate is not None:
                    override_reason = None
                    if brain_engine:
                        try:
                            alert_score, alert_total, _ = _confluence_for(alert_key)
                            override_reason = await brain_engine.check_rewardable_override(
                                alert_key, alert_score, alert_total
                            )              
                        except Exception as e:
                            logger_pair.debug(f"Brain override check failed for {alert_key}: {e}")

                    if override_reason:
                        logger_pair.info(
                            f"[{pair_name}] 🧠 Rewardable override for {alert_key}: "
                            f"WR={failing_rate:.0%} below {cfg.MIN_WIN_RATE:.0%}, but {override_reason}"
                        )
                        alert_extra = f"{alert_extra} | 🧠 {override_reason}"
                        surviving_alerts.append((alert_title, alert_extra, alert_key))
                        continue

                    if cfg.ENABLE_BRAIN and cfg.BRAIN_SHADOW_MODE:
                        alert_score, alert_total, alert_votes = _confluence_for(alert_key)
                        shadow_context = {
                            "rsi_curr": context.get("rsi_curr"),
                            "rsi_adaptive_buy": gr.rsi_adaptive_buy,
                            "rsi_adaptive_sell": gr.rsi_adaptive_sell,
                            "ppo_curr": context.get("ppo_curr"),
                            "ppo_adaptive_threshold": gr.ppo_adaptive_threshold,
                            "buy_wick_ratio": gr.buy_wick_ratio,
                            "sell_wick_ratio": gr.sell_wick_ratio,
                            "adx_val": gr.adx_val,
                            "config_version": hash_config_state(
                                CONFLUENCE_WEIGHTS, cfg.CONFLUENCE_MIN_ABS_SCORE, cfg.CONFLUENCE_MIN_PCT
                            ),
                        }
                        # Redis (shadow tracking)
                        await sdb.record_shadow_pending_outcome(
                            pair_name, alert_key, direction, ts_curr, close_curr,
                            confluence_score=alert_score, confluence_total=alert_total,
                            confluence_votes=alert_votes,
                            context=shadow_context,
                        )
                        # File (long-term brain archive)
       
                    if getattr(cfg, "BRAIN_USE_FILE_STORAGE", False):
                        from outcome_storage import append_outcome
                        append_outcome({
                            "pair": pair_name,
                            "alert_key": alert_key,
                            "direction": direction,
                            "entry_ts": ts_curr,
                            "price": close_curr,
                            "score": alert_score,
                            "total": alert_total,
                            "votes": alert_votes,
                            "context": shadow_context,
                        }, shadow=True)
                    logger_pair.info(
                        f"[{pair_name}] Win-rate filter dropped {alert_key}: {fail_note}"
                    )
                    continue
                surviving_alerts.append((alert_title, alert_extra, alert_key))
            alerts_to_send = surviving_alerts

        coalesced_dedup_key: Optional[str] = None
        if alerts_to_send and cfg.ENABLE_ALERT_COALESCING:
            buy_present = any(ak in BUY_ALERT_KEYS for _, _, ak in alerts_to_send)
            sell_present = any(ak in SELL_ALERT_KEYS for _, _, ak in alerts_to_send)
            direction = "MIXED" if (buy_present and sell_present) else ("BUY" if buy_present else "SELL")
            coalesced_dedup_key = f"coalesced_{direction}"
            should_send = await sdb.check_recent_alert(
                pair_name, coalesced_dedup_key, ts_curr, window_sec=cfg.COALESCE_DEDUP_WINDOW_SEC
            )
            if not should_send:
                logger_pair.debug(
                    f"[{pair_name}] Coalesced {direction} alert deduped (within "
                    f"{cfg.COALESCE_DEDUP_WINDOW_SEC}s) — skipping dispatch"
                )
                alerts_to_send = []
        elif alerts_to_send:
            deduped_alerts = []
            for alert_title, alert_extra, alert_key in alerts_to_send:
                should_send = await sdb.check_recent_alert(pair_name, alert_key, ts_curr)
                if not should_send:
                    logger_pair.debug(f"Alert {alert_key} skipped (dedup window)")
                    continue
                deduped_alerts.append((alert_title, alert_extra, alert_key))
            alerts_to_send = deduped_alerts

        async def _release_dedup_claims() -> None:
            """Releases whichever kind of claim was taken in step 4 above."""
            if coalesced_dedup_key:
                await sdb.release_recent_alert(pair_name, coalesced_dedup_key)
            else:
                for _, _, alert_key in alerts_to_send:
                    await sdb.release_recent_alert(pair_name, alert_key)

        new_alert_activations = []
        for _, _, alert_key in alerts_to_send:
            new_alert_activations.append(
                (f"{pair_name}:{ALERT_KEYS[alert_key]}", "ACTIVE", None)
            )
        async def _refund_alert_budget(n: int) -> None:
            """Undo the optimistic budget reservation when a send does not go out."""
            if n > 0 and alerts_sent_ref is not None and alerts_sent_lock is not None:
                async with alerts_sent_lock:
                    alerts_sent_ref[0] = max(0, alerts_sent_ref[0] - n)

        limit_reached = False

        if alerts_to_send and alerts_sent_ref is not None and alerts_sent_lock is not None:
            async with alerts_sent_lock:
                current_total = alerts_sent_ref[0]
                if current_total >= max_alerts_per_run:
                    limit_reached = True
                else:
                    alerts_sent_ref[0] += len(alerts_to_send)

            if limit_reached:
                logger_pair.warning(
                    f"Global alert limit reached ({current_total}/{max_alerts_per_run}), "
                    f"skipping {len(alerts_to_send)} alerts for {pair_name}"
                )
                if all_state_changes:
                    persist_ok = await sdb.atomic_batch_update(all_state_changes)
                    if not persist_ok:
                        logger_pair.error(
                            f"[{pair_name}] State persistence failed ��� alert state may be inconsistent this run"
                        )
                await _release_dedup_claims()
                return pair_name, {
                    "state": "LIMIT_REACHED",
                    "ts": int(time.time()),
                    "summary": {
                        "alerts": 0,
                        "future_cloud": "green" if cloud_up else "red" if cloud_down else "neutral",
                        "hist_rma": round(hist_curr, 4),
                        "suppression": f"Global limit {max_alerts_per_run} reached"
                    }
                }

        if alerts_to_send:
            budget_refunded = False  # NEW: Flag to prevent double refund
            try:
                if len(alerts_to_send) == 1:
                    title, extra, _ = alerts_to_send[0]
                    msg = build_single_msg(title, pair_name, close_curr, ts_curr, extra, score=confluence_score, total=confluence_total)
                else:
                    items = [(t, e) for t, e, _ in alerts_to_send[:25]]
                    msg = build_batched_msg(pair_name, close_curr, ts_curr, items, score=confluence_score, total=confluence_total)

                if not cfg.DRY_RUN_MODE:
                    reconfirmed = await confirm_candle_unchanged(
                        fetcher, symbol, pair_name, ts_curr, cached_snapshot, reference_time, logger_pair
                    )
                    if reconfirmed is None:
                        logger_pair.warning(
                            f"[{pair_name}] Confirmation inconclusive — alert suppressed this run, "
                            f"dedup key RELEASED so it can retry next run"
                        )
                        await _release_dedup_claims()
                        await _refund_alert_budget(len(alerts_to_send))
                        budget_refunded = True  # Mark as refunded
                        send_success = False
                    elif reconfirmed is False:           
                        logger_pair.warning(
                            f"[{pair_name}] 🔁 Confirmed repaint in send-queue window — "
                            f"alert suppressed, dedup key KEPT to prevent duplicates"
                        )
                        await _refund_alert_budget(len(alerts_to_send))
                        budget_refunded = True  # Mark as refunded
                        send_success = False
                    else:
                        send_success = await telegram_queue.send(msg)

                    if send_success:
                        all_state_changes.extend(new_alert_activations)
                        logger_pair.info(
                            f"🔔🎯🟢 Sent {len(alerts_to_send)} alerts for {pair_name} | "
                            f"Keys: {[ak for _, _, ak in alerts_to_send]}"
                        )
                        if cfg.ENABLE_WIN_RATE_FILTER:
                            async def _record_one(alert_key: str):
                                s, t, v = _confluence_for(alert_key)
                                trigger_context = {
                                    "rsi_curr": context.get("rsi_curr"),
                                    "rsi_adaptive_buy": gr.rsi_adaptive_buy,
                                    "rsi_adaptive_sell": gr.rsi_adaptive_sell,
                                    "ppo_curr": context.get("ppo_curr"),
                                    "ppo_adaptive_threshold": gr.ppo_adaptive_threshold,
                                    "buy_wick_ratio": gr.buy_wick_ratio,
                                    "sell_wick_ratio": gr.sell_wick_ratio,
                                    "adx_val": gr.adx_val,
                                    "config_version": hash_config_state(
                                        CONFLUENCE_WEIGHTS, cfg.CONFLUENCE_MIN_ABS_SCORE, cfg.CONFLUENCE_MIN_PCT
                                    ),
                                    "macro_correlation": macro_shadow.get("correlation") if macro_shadow else None,
                                    "macro_relative_strength": macro_shadow.get("relative_strength") if macro_shadow else None,
                                    "macro_multiplier": macro_shadow.get("multiplier") if macro_shadow else None,
                                    "macro_would_block": macro_shadow.get("would_block") if macro_shadow else None,
                                }
                                # Redis (immediate next-run access)
                                await sdb.record_pending_outcome(
                                    pair_name, alert_key,
                                    "buy" if alert_key in BUY_ALERT_KEYS else "sell",
                                    ts_curr, close_curr,
                                    confluence_score=s, confluence_total=t, confluence_votes=v,
                                    adx_val=adx_val,
                                    context=trigger_context,
                                )
                                # File (long-term brain archive)
                                if getattr(cfg, "BRAIN_USE_FILE_STORAGE", False):
                                    from outcome_storage import append_outcome
                                    append_outcome({
                                        "pair": pair_name,
                                        "alert_key": alert_key,
                                        "direction": "buy" if alert_key in BUY_ALERT_KEYS else "sell",
                                        "entry_ts": ts_curr,
                                        "price": close_curr,
                                        "score": s,
                                        "total": t,
                                        "votes": v,
                                        "adx_val": adx_val,
                                        "context": trigger_context,
                                    })
                            await asyncio.gather(*(_record_one(alert_key) for _, _, alert_key in alerts_to_send))
                    else:
                        # Only refund if not already done
                        if not budget_refunded:
                            await _refund_alert_budget(len(alerts_to_send))
                        logger_pair.error(
                            f"Alert dispatch failed | {pair_name} | "
                            f"State NOT marked ACTIVE, dedup claim retained for retry next run | "
                            f"Budget refunded"
                        )               
                else:
                    # DRY RUN: mark ACTIVE anyway so this run mirrors production dedup/reset behavior
                    all_state_changes.extend(new_alert_activations)
                    logger_pair.info(f"[DRY RUN] Would send: {msg[:100]}...")

            except Exception as e:
                if not budget_refunded:
                    await _refund_alert_budget(len(alerts_to_send))
                logger_pair.error(
                    f"Alert dispatch exception for {pair_name}: {e} | "
                    f"State NOT marked ACTIVE, dedup key retained, budget refunded — "
                    f"will not retry until window expires"
                )

        if all_state_changes:
            await sdb.atomic_batch_update(all_state_changes)

        failed_conditions = [
            name for name, val in [
                ("buy_common", buy_common),
                ("sell_common", sell_common),
            ] if not val
        ]
        reasons = []
        if not alerts_to_send:
            if not buy_common and not sell_common:
                reasons.append("Trend filter blocked")
            
            if context.get("pivot_suppressions"):
                reasons.extend(context["pivot_suppressions"])

            if ppo_prev <= 0 and ppo_curr > 0 and not buy_common:
                if not base_buy_trend:
                    reasons.append("PPO>0 blocked: base_buy_trend=False")
                elif not confirmation_buy:
                    reasons.append("PPO>0 blocked: confirmation_buy=False (future cloud)")
                elif not is_valid_for_buy:
                    reasons.append("PPO>0 blocked: Knox rejected candle (wick/color/timing)")
                else:
                    reasons.append(
                        f"PPO>0 blocked: market filter "
                        f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                        f"rvol_ok={rvol_ok})"
                    )           
        
            if ppo_prev >= 0 and ppo_curr < 0 and not sell_common:
                if not base_sell_trend:
                    reasons.append("PPO<0 blocked: base_sell_trend=False")
                elif not confirmation_sell:
                    reasons.append("PPO<0 blocked: confirmation_sell=False (future cloud)")
                elif not is_valid_for_sell:
                    reasons.append("PPO<0 blocked: Knox rejected candle (wick/color/timing)")
                else:
                    reasons.append(
                        f"PPO<0 blocked: market filter "
                        f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                        f"rvol_ok={rvol_ok})"
                    )
       
            if ppo_prev <= ppo_adaptive_threshold and ppo_curr > ppo_adaptive_threshold and not buy_common:
                if not base_buy_trend:
                    reasons.append("PPO>+adapt blocked: base_buy_trend=False")
                elif not confirmation_buy:
                    reasons.append("PPO>+adapt blocked: confirmation_buy=False (future cloud)")
                elif not is_valid_for_buy:
                    reasons.append("PPO>+adapt blocked: Knox rejected candle")
                else:
                    reasons.append(
                        f"PPO>+{ppo_adaptive_threshold:.3f} blocked: market filter "
                        f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                        f"rvol_ok={rvol_ok})"             
                    )
        
            if ppo_prev >= -ppo_adaptive_threshold and ppo_curr < -ppo_adaptive_threshold and not sell_common:
                if not base_sell_trend:
                    reasons.append("PPO<-adapt blocked: base_sell_trend=False")
                elif not confirmation_sell:
                    reasons.append("PPO<-adapt blocked: confirmation_sell=False (future cloud)")
                elif not is_valid_for_sell:
                    reasons.append("PPO<-adapt blocked: Knox rejected candle")
                else:
                    reasons.append(
                        f"PPO<-{ppo_adaptive_threshold:.3f} blocked: market filter "
                        f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                        f"rvol_ok={rvol_ok})"
                    )      

            if rsi_prev <= rsi_ema_prev and rsi_curr > rsi_ema_curr:
                if rsi_curr >= rsi_adaptive_buy:
                    reasons.append(f"RSI>EMA5 blocked: RSI={rsi_curr:.2f} ≥ cap {rsi_adaptive_buy:.1f}")
                elif ppo_gate_curr >= Constants.PPO_RSI_GUARD_BUY:
                    reasons.append(f"RSI>EMA5 blocked: PPO={ppo_gate_curr:.2f} ≥ guard {Constants.PPO_RSI_GUARD_BUY}")
                elif not buy_common:
                    if not base_buy_trend:
                        reasons.append("RSI>EMA5 blocked: base_buy_trend=False")
                    elif not confirmation_buy:
                        reasons.append("RSI>EMA5 blocked: confirmation_buy=False (future cloud)")
                    elif not is_valid_for_buy:
                        reasons.append("RSI>EMA5 blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"RSI>EMA5 blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )

            if rsi_prev >= rsi_ema_prev and rsi_curr < rsi_ema_curr:
                if rsi_curr <= rsi_adaptive_sell:
                    reasons.append(f"RSI<EMA5 blocked: RSI={rsi_curr:.2f} ≤ cap {rsi_adaptive_sell:.1f}")
                elif ppo_gate_curr <= Constants.PPO_RSI_GUARD_SELL:
                    reasons.append(f"RSI<EMA5 blocked: PPO={ppo_gate_curr:.2f} ≤ guard {Constants.PPO_RSI_GUARD_SELL}")
                elif not sell_common:
                    if not base_sell_trend:
                        reasons.append("RSI<EMA5 blocked: base_sell_trend=False")
                    elif not confirmation_sell:
                        reasons.append("RSI<EMA5 blocked: confirmation_sell=False (future cloud)")
                    elif not is_valid_for_sell:
                        reasons.append("RSI<EMA5 blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"RSI<EMA5 blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )
       
            if cfg.ENABLE_VWAP and vwap_available:
                if close_prev <= vwap_prev and close_curr > vwap_curr and not buy_common:
                    if not base_buy_trend:
                        reasons.append("VWAP up-cross blocked: base_buy_trend=False")
                    elif not confirmation_buy:
                        reasons.append("VWAP up-cross blocked: confirmation_buy=False")
                    elif not is_valid_for_buy:
                        reasons.append("VWAP up-cross blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"VWAP up-cross blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )
            
                if close_prev >= vwap_prev and close_curr < vwap_curr and not sell_common:
                    if not base_sell_trend:
                        reasons.append("VWAP down-cross blocked: base_sell_trend=False")
                    elif not confirmation_sell:
                        reasons.append("VWAP down-cross blocked: confirmation_sell=False")
                    elif not is_valid_for_sell:
                        reasons.append("VWAP down-cross blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"VWAP down-cross blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )

            if cfg.ENABLE_PPO_GATE:
                if not ppo_gate_ok_buy:
                    reasons.append(f"PPO Gate buy: Gate({ppo_gate_curr:.2f}) <= Signal({ppo_gate_sig_curr:.2f})")
                if not ppo_gate_ok_sell:
                    reasons.append(f"PPO Gate sell: Gate({ppo_gate_curr:.2f}) >= Signal({ppo_gate_sig_curr:.2f})")

            if cfg.ENABLE_TK_CONVERSION_CROSS:
                if close_prev <= tk_conversion_prev and close_curr > tk_conversion_curr and not buy_common:
                    if not base_buy_trend:
                        reasons.append("Conversion up-cross blocked: base_buy_trend=False")
                    elif not confirmation_buy:
                        reasons.append("Conversion up-cross blocked: confirmation_buy=False")
                    elif not is_valid_for_buy:
                        reasons.append("Conversion up-cross blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"Conversion up-cross blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )

                if close_prev >= tk_conversion_prev and close_curr < tk_conversion_curr and not sell_common:
                    if not base_sell_trend:
                        reasons.append("Conversion down-cross blocked: base_sell_trend=False")
                    elif not confirmation_sell:
                        reasons.append("Conversion down-cross blocked: confirmation_sell=False")
                    elif not is_valid_for_sell:
                        reasons.append("Conversion down-cross blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"Conversion down-cross blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )

            if cfg.ENABLE_KIJUN_CROSS:
                if close_prev <= tk_base_prev and close_curr > tk_base_curr and not buy_common:
                    if not base_buy_trend:
                        reasons.append("Kijun up-cross blocked: base_buy_trend=False")
                    elif not confirmation_buy:
                        reasons.append("Kijun up-cross blocked: confirmation_buy=False")
                    elif not is_valid_for_buy:
                        reasons.append("Kijun up-cross blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"Kijun up-cross blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )

                if close_prev >= tk_base_prev and close_curr < tk_base_curr and not sell_common:
                    if not base_sell_trend:
                        reasons.append("Kijun down-cross blocked: base_sell_trend=False")
                    elif not confirmation_sell:
                        reasons.append("Kijun down-cross blocked: confirmation_sell=False")
                    elif not is_valid_for_sell:
                        reasons.append("Kijun down-cross blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"Kijun down-cross blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )

            if cfg.ENABLE_HIST_RMA:
                if buy_common and not hist_reversal_buy:
                    if np.isnan(hist_curr):
                        reasons.append("Hist RMA buy: NaN")
                    elif hist_curr <= 0:
                        reasons.append(f"Hist RMA buy: hist_curr={hist_curr:.2f} <= 0")
                    elif not (hist_m3 > hist_m2 > hist_m1):
                        reasons.append(f"Hist RMA buy: sequence not rising ({hist_m3:.2f} > {hist_m2:.2f} > {hist_m1:.2f})")
                    elif not (hist_curr > hist_m1):
                        reasons.append(f"Hist RMA buy: no acceleration ({hist_curr:.2f} <= {hist_m1:.2f})")
                if sell_common and not hist_reversal_sell:
                    if np.isnan(hist_curr):
                        reasons.append("Hist RMA sell: NaN")
                    elif hist_curr >= 0:
                        reasons.append(f"Hist RMA sell: hist_curr={hist_curr:.2f} >= 0")
                    elif not (hist_m3 < hist_m2 < hist_m1):
                        reasons.append(f"Hist RMA sell: sequence not falling ({hist_m3:.2f} < {hist_m2:.2f} < {hist_m1:.2f})")
                    elif not (hist_curr < hist_m1):
                        reasons.append(f"Hist RMA sell: no acceleration ({hist_curr:.2f} >= {hist_m1:.2f})")

            if cfg.RSI_GUARD_ENABLED:
                if not rsi_guard_ok_buy:
                    reasons.append(f"RSI Guard buy: RSI({rsi_guard_smooth_curr:.2f}) <= EMA({rsi_guard_ema_curr:.2f})")
                if not rsi_guard_ok_sell:
                    reasons.append(f"RSI Guard sell: RSI({rsi_guard_smooth_curr:.2f}) >= EMA({rsi_guard_ema_curr:.2f})")

            if cfg.RMA_CLOUD_ENABLED:
                if not rma_cloud_ok_buy:
                    reasons.append(f"RMA Cloud buy: RMA{cfg.RMA_CLOUD_FAST_PERIOD}({rma_cloud_fast_curr:.2f}) <= RMA{cfg.RMA_50_PERIOD}({rma50_15_val:.2f})")
                if not rma_cloud_ok_sell:
                    reasons.append(f"RMA Cloud sell: RMA{cfg.RMA_CLOUD_FAST_PERIOD}({rma_cloud_fast_curr:.2f}) >= RMA{cfg.RMA_50_PERIOD}({rma50_15_val:.2f})")
            if cfg.ENABLE_OB_GATE:
                if gr.ob_gate_ok_buy is False:
                    reasons.append(f"OB buy: {gr.ob_gate_reason or 'zone touched, no reversal confirmed'}")
                if gr.ob_gate_ok_sell is False:
                    reasons.append(f"OB sell: {gr.ob_gate_reason or 'zone touched, no reversal confirmed'}")
            if cfg.ICHIMOKU_CLOUD_ENABLED:
                if not ichimoku_gate_ok_buy:
                    reasons.append(f"Ichimoku Cloud buy: price not above cloud / future not green (vote)")
                if not ichimoku_gate_ok_sell:
                    reasons.append(f"Ichimoku Cloud sell: price not below cloud / future not red (vote)")

            if not cloud_group_ok_buy:
                reasons.append("Cloud group buy: need ANY enabled vote true (Ichimoku/RMA cloud) — all disagree or abstain")
            if not cloud_group_ok_sell:
                reasons.append("Cloud group sell: need ANY enabled vote true (Ichimoku/RMA cloud) — all disagree or abstain")
            if not oscillator_group_ok_buy:
                reasons.append(f"Oscillator group buy: need {Constants.OSCILLATOR_GROUP_MIN_VOTES}-of-3 (PPO/RSI/TK) — not met")
            if not oscillator_group_ok_sell:
                reasons.append(f"Oscillator group sell: need {Constants.OSCILLATOR_GROUP_MIN_VOTES}-of-3 (PPO/RSI/TK) — not met")

            logger_pair.debug(f"😒 {pair_name} | Suppression: {', '.join(reasons)}") 

        return pair_name, {
            "state": "ALERT_SENT" if alerts_to_send else "NO_SIGNAL",
            "ts": int(time.time()),
            "summary": {
                "alerts": len(alerts_to_send),
                "future_cloud": "green" if cloud_up else "red" if cloud_down else "neutral",
                "hist_rma": round(hist_curr, 4), 
                "suppression": ", ".join(failed_conditions + reasons) if (failed_conditions or reasons) else "No conditions met"
            }
        }
    except asyncio.CancelledError:
        logger_pair.warning(f"Evaluation cancelled for {pair_name}")
        raise
    except RuntimeError as e:
        logger_pair.critical(f"🚨 INVARIANT VIOLATION in {pair_name}: {e}")
        return pair_name, {
            "state": "INVARIANT_VIOLATION",
            "ts": int(time.time()),
            "summary": {
                "alerts": 0,
                "future_cloud": "neutral",
                "hist_rma": 0.0,
                "error": str(e)
            }
        }
    except Exception as e:
        logger_pair.exception(
            f"❌ Error in _apply_and_dispatch_alerts for {pair_name}: {e} | Correlation: {correlation_id}"
        )
        return None
