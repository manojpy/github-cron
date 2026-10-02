"""Dispatcher: ordering by confluence %, same-underlying note, bias footer."""
import asyncio
import logging
from alerts import AlertPayload, dispatch_combined_alerts
from bot_config import BiasContext


def _bias(up, down, neutral):
    n = 30
    return BiasContext(up_count=round(up * n), down_count=round(down * n),
                       neutral_count=round(neutral * n), total_pairs=n,
                       up_pct=up, down_pct=down, neutral_pct=neutral)


UP, DOWN = _bias(0.6, 0.2, 0.2), _bias(0.2, 0.57, 0.23)


class _Tg:
    def __init__(self):
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)
        return True


class _Sdb:
    async def atomic_batch_update(self, c):
        return True

    async def set_last_processed_candle_ts(self, *a):
        return None

    async def release_recent_alert(self, *a):
        return None


def _p(pair, direction, score, total, verdict=None):
    return AlertPayload(pair_name=pair, direction=direction, score=score, total=total,
                        msg_body=f"BODY-{pair}", dedup_keys=["k"], state_changes=[],
                        budget_count=1, ts=1_790_826_300, alert_keys=["vwap_up"],
                        verdict=verdict)


def _send(payloads, bias):
    tg = _Tg()
    asyncio.run(dispatch_combined_alerts(payloads, tg, bias, _Sdb(), [0], asyncio.Lock(), 50,
                                         logging.getLogger("t")))
    return "\n".join(tg.sent)


def _order(text, names):
    return [n for n in sorted(names, key=lambda n: text.index(f"BODY-{n}"))]


def test_orders_by_confluence_percent_not_raw_score():
    # raw score says A(26/30=87%) > B(25/27=93%); percent must put B first
    ps = [_p("AAAUSD", "buy", 26, 30), _p("BBBUSD", "buy", 25, 27), _p("CCCUSD", "sell", 28, 29)]
    text = _send(ps, UP)
    assert _order(text, ["AAAUSD", "BBBUSD", "CCCUSD"]) == ["BBBUSD", "AAAUSD", "CCCUSD"]


def test_negative_bias_puts_sells_first():
    ps = [_p("AAAUSD", "buy", 26, 30), _p("CCCUSD", "sell", 20, 29)]
    text = _send(ps, DOWN)
    assert _order(text, ["AAAUSD", "CCCUSD"]) == ["CCCUSD", "AAAUSD"]


def test_same_underlying_note_only_on_later_same_direction_member():
    ps = [_p("PAXGUSD", "sell", 24, 27), _p("XAUTUSD", "sell", 22, 27), _p("ETHUSD", "sell", 20, 27)]
    text = _send(ps, DOWN)
    assert text.count("Same underlying as PAXGUSD") == 1
    assert text.index("Same underlying") > text.index("BODY-XAUTUSD")
    assert text.index("Same underlying") < text.index("BODY-ETHUSD")


def test_opposite_directions_of_a_group_are_not_flagged():
    ps = [_p("PAXGUSD", "sell", 24, 27), _p("XAUTUSD", "buy", 22, 27)]
    assert "Same underlying" not in _send(ps, DOWN)


def test_footer_has_candle_open_time_and_bias():
    text = _send([_p("AAAUSD", "buy", 24, 27)], DOWN)
    assert "⏰" in text and "📆" in text and "Bias" in text


def test_basket_note_when_three_same_direction_takes():
    ps = [_p(n, "buy", 26, 27, "TAKE") for n in ("AAAUSD", "BBBUSD", "CCCUSD")] + [_p("DDDUSD", "buy", 20, 27, "WATCH")]
    text = _send(ps, UP)
    assert text.count("size them as ONE basket") == 1 and "3 BUY TAKEs" in text


def test_no_basket_note_below_threshold_or_for_watch():
    two = [_p("AAAUSD", "buy", 26, 27, "TAKE"), _p("BBBUSD", "buy", 25, 27, "TAKE")]
    assert "basket" not in _send(two, UP)
    watch = [_p(n, "buy", 26, 27, "WATCH") for n in ("AAAUSD", "BBBUSD", "CCCUSD")]
    assert "basket" not in _send(watch, UP)
