"""Tier report: grouping maths, verdict wording, and an end-to-end archive read."""
import json

import confluence_tier_report as ctr
from outcome_storage import OUTCOME_SCHEMA_VERSION


def _row(conf, win, ev, direction="buy"):
    return {"conf_pct": conf, "win": win, "net_pnl_pct": ev, "direction": direction}


def test_groups_rows_into_tiers_and_directions():
    rows = [_row(96, True, 1.0), _row(92, False, -0.5), _row(92, True, 0.8, "sell"),
            _row(86, True, 0.2), _row(70, False, -1.0)]
    rep = ctr.tier_report(rows, bar=90)
    assert rep["tiers"]["95-100%"]["n"] == 1 and rep["tiers"]["90-95%"]["n"] == 2
    assert rep["tiers"]["<80%"]["n"] == 1
    assert rep["by_direction"]["sell"]["90-95%"]["n"] == 1
    assert rep["at_or_above_bar"]["n"] == 3
    assert abs(rep["tiers"]["90-95%"]["net_ev"] - 0.15) < 1e-9


def test_verdict_needs_enough_data_then_judges_edge():
    few = ctr.tier_report([_row(95, True, 1.0)] * 5)
    assert few["verdict"].startswith("NOT ENOUGH DATA")
    good = ctr.tier_report([_row(95, True, 0.9)] * 20 + [_row(95, False, -0.4)] * 15)
    assert good["verdict"].startswith("SUPPORTED")
    bad = ctr.tier_report([_row(95, True, 0.2)] * 10 + [_row(95, False, -1.0)] * 25)
    assert bad["verdict"].startswith("NOT SUPPORTED")


def test_empty_input_does_not_crash():
    rep = ctr.tier_report([])
    assert rep["n_total"] == 0 and "NOT ENOUGH DATA" in rep["verdict"]
    assert "Confluence tier report" in ctr.render(rep)


def test_reads_the_real_archive_format(tmp_path, capsys):
    import time
    day = time.strftime("%Y-%m-%d", time.gmtime())
    d = tmp_path / "outcomes"
    d.mkdir()
    base = dict(pair="AAVEUSD", alert_key="vwap_up", direction="buy", entry_ts=int(time.time()) - 3600,
                score=26, total=27, pct_move=1.2, win=True, session="asia", votes={}, context={},
                mae=0.2, mfe=1.5, close_win=True, mfe_win=True, mae_loss=False, tp_first=True,
                outcome_reason="tp", bonus_win=False, rr_achieved=1.5, win_weight=1.0,
                net_pnl_pct=1.0, realized_cost_pct=0.1, adx_val=30.0, schema_version=OUTCOME_SCHEMA_VERSION)
    with open(d / f"{day}.jsonl", "w") as f:
        for i in range(3):
            f.write(json.dumps({**base, "outcome_id": f"id{i}", "sid": f"s{i}"}) + "\n")
    assert ctr.main(["--data-dir", str(tmp_path), "--days", "5", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_total"] == 3 and out["tiers"]["95-100%"]["n"] == 3 and out["tiers"]["90-95%"]["n"] == 0
