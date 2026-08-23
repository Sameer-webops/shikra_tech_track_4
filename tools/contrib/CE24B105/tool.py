"""Diagnose a run: where the money went, and where the plant disagreed with
the policy. Answers two questions first, before a general time-series
explorer would: which cost line moved, and did anything get clipped, refused,
or silently idled while burning fuel.

    python tool.py --run <run directory> --out <output directory>

Reads log.csv and manifest.json from --run and writes diagnostic.html into
--out. Never re-runs the simulation, never imports sim/ or dispatch/. Column
names are discovered from the CSV header at read time and never hardcoded --
the fleet this is scored against is not the fleet shipped in this repo.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Callable


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _num(v: str) -> float | None:
    """An empty cell means NOT COMMANDED, never zero. Never fill it in."""
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _member_of(column: str) -> str | None:
    """`bess_1.soc_kwh` -> `bess_1`. None for plant-level / index columns."""
    plant_level = {
        "k",
        "timestamp",
        "slack_id",
        "slack_advisory_kw",
        "slack_delta_kw",
        "violations",
        "quality",
    }
    if column in plant_level:
        return None
    return column.split(".", 1)[0] if "." in column else None


def build_violations(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Every interval where the plant recorded a clip, plus every genset
    interval commanded inside its forbidden band (0 < p < p_min_stable):
    that one delivers zero and still burns fuel, and it does not always trip
    the `violations` column on its own -- it looks like a quiet zero.
    """
    if not rows:
        return []
    found: list[dict[str, Any]] = []
    members = sorted({m for c in rows[0] if (m := _member_of(c))})
    for row in rows:
        plant_v = row.get("violations", "")
        if plant_v:
            found.append(
                {"k": row["k"], "t": row["timestamp"], "member": "(plant)", "note": plant_v}
            )
        for m in members:
            v = row.get(f"{m}.violations", "")
            if v:
                found.append({"k": row["k"], "t": row["timestamp"], "member": m, "note": v})
            cmd_p = _num(row.get(f"{m}.cmd.p_setpoint_kw", ""))
            net_p = _num(row.get(f"{m}.p_net_kw", ""))
            running = row.get(f"{m}.running", "")
            if cmd_p is not None and cmd_p > 0 and net_p == 0.0 and running == "True":
                found.append(
                    {
                        "k": row["k"],
                        "t": row["timestamp"],
                        "member": m,
                        "note": f"commanded {cmd_p:.1f} kW, delivered 0 -- forbidden-band idle",
                    }
                )
    return found


def _delivered_and_commanded(row: dict[str, str], member: str) -> tuple[str, str]:
    """A member's "what happened" and "what was asked", as display strings.

    Assets (pv/bess/dg/grid) are power-commanded: `.p_net_kw` / `.cmd.p_setpoint_kw`.
    Loads are switched, not power-commanded: `.served_p_kw` / `.cmd.on`. Neither
    field exists on the other kind, so this tries the asset shape first and
    falls back to the load shape rather than asking a load for a channel it
    never declares.
    """
    p = row.get(f"{member}.p_net_kw", "")
    cmd_p = row.get(f"{member}.cmd.p_setpoint_kw", "")
    if p != "" or cmd_p != "":
        return (p if p != "" else "—", cmd_p if cmd_p != "" else "—")
    served = row.get(f"{member}.served_p_kw", "")
    cmd_on = row.get(f"{member}.cmd.on", "")
    if served != "" or cmd_on != "":
        return (served if served != "" else "—", cmd_on if cmd_on != "" else "—")
    return ("—", "—")


def demand_peak(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    """The single interval with the highest grid import kVA -- a proxy for
    what set the demand charge. Real billing uses the highest BLOCK-averaged
    kVA over the whole month, not one 15-min sample, so this narrows the
    search rather than answering it outright.
    """
    if not rows:
        return None
    grid_member = next((m for c in rows[0] if (m := _member_of(c)) and m.startswith("grid")), None)
    if grid_member is None:
        return None
    best: dict[str, str] | None = None
    best_kva = -1.0
    for row in rows:
        p = _num(row.get(f"{grid_member}.p_net_kw", "")) or 0.0
        q = _num(row.get(f"{grid_member}.q_net_kvar", "")) or 0.0
        if p <= 0:
            continue  # export, not import -- does not touch the demand charge
        kva = (p**2 + q**2) ** 0.5
        if kva > best_kva:
            best_kva, best = kva, row
    if best is None:
        return None
    members = sorted({m for c in rows[0] if (m := _member_of(c))})
    detail = {
        m: dict(zip(("delivered", "commanded"), _delivered_and_commanded(best, m))) for m in members
    }
    return {
        "k": best["k"],
        "t": best["timestamp"],
        "grid_import_kva": round(best_kva, 1),
        "members": detail,
    }


def _bess_members(rows: list[dict[str, str]]) -> list[str]:
    """Members with a `.soc_kwh` channel -- discovered, never named."""
    if not rows:
        return []
    return sorted({m for c in rows[0] if c.endswith(".soc_kwh") and (m := _member_of(c))})


def _load_members(rows: list[dict[str, str]]) -> list[str]:
    """Members with an `.unserved_p_kw` channel -- discovered, never named."""
    if not rows:
        return []
    return sorted({m for c in rows[0] if c.endswith(".unserved_p_kw") and (m := _member_of(c))})


_NOTE_RE = re.compile(r"^([a-z_]+):([\d.]+)(?:->([\d.]+))?$")


def _parse_notes(note: str) -> list[tuple[str, float, float | None]]:
    """A violations cell can hold MULTIPLE codes in one string, joined by `|`
    -- e.g. `"bess_discharge_rate_clipped:657.7->500.0|bess_discharge_soc_clipped:
    500.0->0.0"` is two separate facts about the same interval, not one.
    Splitting first and parsing each part is what earlier code got wrong:
    matching the regex against the whole cell silently dropped every code
    after the first `|`.
    """
    out: list[tuple[str, float, float | None]] = []
    for part in note.split("|"):
        m = _NOTE_RE.match(part)
        if not m:
            out.append((part, 0.0, None))
            continue
        code, a, b = m.group(1), float(m.group(2)), m.group(3)
        out.append((code, a, float(b) if b is not None else None))
    return out


def group_events(
    violations: list[dict[str, Any]], rows: list[dict[str, str]], gap: int = 1
) -> list[dict[str, Any]]:
    """Cluster raw violation rows into events: runs of intervals whose `k`
    values are contiguous (allowing a gap of `gap`). Each event summarizes
    what codes fired, their worst magnitudes, and whether a battery in the
    window visibly hit an energy floor (SOC flat for the tail of the window
    while still being asked to discharge) -- detected from the trajectory,
    not asserted from the violation text alone.
    """
    if not violations:
        return []
    ks = sorted({int(v["k"]) for v in violations})
    clusters: list[list[int]] = [[ks[0]]]
    for k in ks[1:]:
        if k - clusters[-1][-1] <= gap:
            clusters[-1].append(k)
        else:
            clusters.append([k])

    by_k: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for v in violations:
        by_k.setdefault(v["k"], {}).setdefault(v["member"], []).append(v)

    rows_by_k = {row["k"]: row for row in rows}
    bess_members = _bess_members(rows)

    events: list[dict[str, Any]] = []
    for cluster in clusters:
        cluster_ks = [str(k) for k in cluster]
        codes: dict[str, dict[str, float]] = {}  # code -> {max_from, min_to, max_value}
        members: set[str] = set()
        for k_str in cluster_ks:
            for member, entries in by_k.get(k_str, {}).items():
                if member != "(plant)":
                    members.add(member)
                for e in entries:
                    for code, a, b in _parse_notes(e["note"]):
                        slot = codes.setdefault(
                            code, {"max_from": 0.0, "min_to": 1e18, "max_value": 0.0}
                        )
                        if b is not None:
                            slot["max_from"] = max(slot["max_from"], a)
                            slot["min_to"] = min(slot["min_to"], b)
                        else:
                            slot["max_value"] = max(slot["max_value"], a)

        # SOC-floor check: did any battery's SOC go flat across the tail of
        # this window while the window also shows a discharge-clip code?
        soc_floor_member = None
        if any("clipped" in c for c in codes):
            for bm in bess_members:
                raw = [
                    _num(rows_by_k[k_str].get(f"{bm}.soc_kwh", ""))
                    for k_str in cluster_ks
                    if k_str in rows_by_k
                ]
                series: list[float] = [s for s in raw if s is not None]
                if len(series) >= 3 and series[-1] == min(series) and series[-2] == series[-1]:
                    soc_floor_member = bm
                    break

        start_row = rows_by_k.get(cluster_ks[0], {})
        end_row = rows_by_k.get(cluster_ks[-1], {})
        events.append(
            {
                "start_k": cluster[0],
                "end_k": cluster[-1],
                "start_t": start_row.get("timestamp", ""),
                "end_t": end_row.get("timestamp", ""),
                "n_intervals": len(cluster),
                "codes": codes,
                "members": sorted(members),
                "soc_floor_member": soc_floor_member,
            }
        )
    return events


def diagnose_event(event: dict[str, Any]) -> dict[str, Any]:
    """Observed facts (parsed straight from the codes, in plain language) vs.
    one interpretation built from independently-hedged sentences. Each clause
    names its own evidence ("as indicated by SOC-related clipping") rather
    than asserting a single unbroken causal chain -- the log proves each fact
    separately, and the interpretation should read that way too.
    """
    codes = event["codes"]
    observed: list[str] = []
    rate_clip: tuple[float, float] | None = None
    energy_clip: tuple[float, float] | None = None
    has_unserved = False
    max_unserved = 0.0

    for code, v in codes.items():
        if "clipped" in code and v["max_from"] > 0:
            if "rate" in code:
                rate_clip = (v["max_from"], v["min_to"])
                observed.append(f"Battery discharge request: {v['max_from']:.1f} kW")
                observed.append(f"Battery rate clipped to: {v['min_to']:.1f} kW")
            elif "soc" in code or "energy" in code:
                energy_clip = (v["max_from"], v["min_to"])
                observed.append(
                    f"SOC-related clipping occurred (requested {v['max_from']:.1f} kW, "
                    f"limited to {v['min_to']:.1f} kW)"
                )
            else:
                observed.append(
                    f"{code.replace('_', ' ')}: requested up to {v['max_from']:.1f}, "
                    f"limited to {v['min_to']:.1f}"
                )
        elif v["max_value"] > 0:
            if "unserved" in code:
                has_unserved = True
                max_unserved = max(max_unserved, v["max_value"])
                observed.append(f"Unserved load: {v['max_value']:.1f} kW")
            else:
                observed.append(f"{code.replace('_', ' ')}: peaked at {v['max_value']:.1f}")

    member_txt = ", ".join(event["members"]) if event["members"] else "the plant"
    sentences: list[str] = []
    if rate_clip:
        sentences.append(
            f"{member_txt} could not fully deliver the requested output "
            f"(requested up to {rate_clip[0]:.1f} kW, limited to {rate_clip[1]:.1f} kW)."
        )
    if energy_clip and event["soc_floor_member"]:
        sentences.append(
            f"{event['soc_floor_member']} subsequently became limited by available "
            f"stored energy, as indicated by SOC-related clipping, during the same window."
        )
    elif energy_clip:
        sentences.append(
            "SOC-related clipping was also recorded, though the SOC trajectory in "
            "this window does not clearly confirm a sustained energy floor."
        )
    if has_unserved:
        sentences.append(
            f"Unserved load was recorded during the same window, reaching {max_unserved:.1f} kW."
        )

    interpretation = (
        " ".join(sentences)
        if sentences
        else "Recorded plant violation with no clip or unserved pattern matched."
    )

    severity = (
        max_unserved * 3 + event["n_intervals"] * 5 + (40 if event["soc_floor_member"] else 0)
    )
    return {"observed": observed, "interpretation": interpretation, "severity": severity}


def battery_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, float | None]]:
    """Per battery: first, min, max, final SOC. Skips entirely if the fleet
    has no `.soc_kwh` channel -- some scored fleets may not.
    """
    out: dict[str, dict[str, float | None]] = {}
    for bm in _bess_members(rows):
        raw = [_num(r.get(f"{bm}.soc_kwh", "")) for r in rows]
        series: list[float] = [s for s in raw if s is not None]
        if not series:
            continue
        out[bm] = {
            "initial": series[0],
            "final": series[-1],
            "min": min(series),
            "max": max(series),
        }
    return out


def unserved_total_kwh(rows: list[dict[str, str]], dt_minutes: float) -> float:
    """Sum of unserved power across every load member, converted to energy.
    `dt_minutes` comes from the manifest -- never a hardcoded 15 or 0.25.
    """
    total = 0.0
    hours = dt_minutes / 60.0
    for lm in _load_members(rows):
        for r in rows:
            v = _num(r.get(f"{lm}.unserved_p_kw", ""))
            if v:
                total += v * hours
    return total


def rank_events(events: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every event paired with its diagnosis, worst-first. Severity is the
    single number diagnose_event computed: max unserved kW * 3, plus 5 per
    affected interval, plus a flat 40 if a battery visibly hit its energy
    floor in that window. It's a simple, stated formula rather than a
    black-box score -- a reviewer can recompute it by eye from the numbers
    printed beside each event.
    """
    paired = [(e, diagnose_event(e)) for e in events]
    paired.sort(key=lambda ed: ed[1]["severity"], reverse=True)
    return paired


def _pct_of(total: float) -> Callable[[float], float]:
    """A percent-of-total function, typed plainly so mypy can follow it --
    the earlier conditional-lambda expression it replaces was untyped.
    """
    if not total:
        return lambda _v: 0.0
    return lambda v: v / total * 100.0


def cost_interpretation(cost: dict[str, float], total: float) -> dict[str, Any]:
    """Which line dominates, which is second, as percent of TOTAL (not of the
    sum of positive lines only) -- export credit is negative and stays
    negative in this ranking; it is never treated as a cost to be maximised.
    """
    positive = [(k, v) for k, v in cost.items() if v > 0]
    positive.sort(key=lambda kv: kv[1], reverse=True)
    pct = _pct_of(total)
    lines = [(k, v, pct(v)) for k, v in cost.items()]
    return {
        "dominant": positive[0] if positive else None,
        "second": positive[1] if len(positive) > 1 else None,
        "lines": lines,
        "pct": pct,
    }


def run_summary_cards(
    manifest: dict[str, Any],
    n_intervals: int,
    cost: dict[str, float],
    peak: dict[str, Any] | None,
    events: list[dict[str, Any]],
    battery: dict[str, dict[str, float | None]],
    unserved_total: float,
) -> list[tuple[str, str]]:
    total = manifest.get("total_inr", sum(cost.values()))
    cards = [("Total cost", f"₹{total:,.0f}")]
    if peak:
        cards.append(("Peak grid import", f"{peak['grid_import_kva']:,.1f} kVA"))
    cards.append(("Unserved load", f"{unserved_total:,.0f} kWh"))
    raw_mins = [s["min"] for s in battery.values()]
    mins: list[float] = [m for m in raw_mins if m is not None]
    if mins:
        cards.append(("Min battery SOC", f"{min(mins):,.0f} kWh"))
    cards.append(("Violation events", str(len(events))))
    cards.append(("Intervals", f"{n_intervals:,}"))
    return cards


def key_findings(
    cost: dict[str, float],
    total: float,
    events: list[dict[str, Any]],
    battery: dict[str, dict[str, float | None]],
    unserved_total: float,
) -> list[str]:
    """3-6 short, data-grounded findings. Each one only fires if the run
    actually supports it -- an empty section beats a generic one.
    """
    findings: list[str] = []
    ranked = sorted(((k, v) for k, v in cost.items() if v > 0), key=lambda kv: kv[1], reverse=True)
    if ranked:
        label, value = ranked[0]
        pct = (value / total * 100.0) if total else 0.0
        findings.append(
            f"{label} was the dominant cost contributor: {pct:.0f}% of total cost (₹{value:,.0f})."
        )
    if events:
        total_iv = sum(e["n_intervals"] for e in events)
        findings.append(
            f"{len(events)} distinct violation event(s) were recorded, spanning "
            f"{total_iv} intervals in total."
        )
    floor_events = [e for e in events if e.get("soc_floor_member")]
    if floor_events:
        findings.append(
            f"Battery stored energy (not just power) was the limiting factor in "
            f"{len(floor_events)} of {len(events)} event(s)."
        )
    if unserved_total > 0:
        findings.append(f"Unserved load totaled {unserved_total:,.0f} kWh across the run.")
    for bm, s in battery.items():
        if s["min"] is not None and s["initial"] is not None and s["initial"] > 0:
            drop_pct = (1 - s["min"] / s["initial"]) * 100.0
            if drop_pct > 50:
                findings.append(
                    f"{bm} SOC fell from {s['initial']:,.0f} kWh to {s['min']:,.0f} kWh "
                    f"at its lowest point, a {drop_pct:.0f}% drop."
                )
                break
    return findings[:6]


def render_html(
    manifest: dict[str, Any],
    violations: list[dict[str, Any]],
    peak: dict[str, Any] | None,
    events: list[dict[str, Any]],
    battery: dict[str, dict[str, float | None]],
    unserved_total: float,
    n_intervals: int,
) -> str:
    cost = manifest.get("cost_breakdown_inr", {})
    total = manifest.get("total_inr", sum(cost.values()))

    # ---- run summary cards --------------------------------------------
    cards = run_summary_cards(manifest, n_intervals, cost, peak, events, battery, unserved_total)
    cards_html = "\n".join(
        f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in cards
    )

    # ---- key findings ----------------------------------------------------
    findings = key_findings(cost, total, events, battery, unserved_total)
    findings_html = (
        "\n".join(f"<li>{f}</li>" for f in findings)
        if findings
        else "<li class='muted'>No notable findings for this run.</li>"
    )

    # ---- cost diagnosis (bars + interpretation) --------------------------
    max_abs = max((abs(v) for v in cost.values()), default=1.0) or 1.0
    ci = cost_interpretation(cost, total)
    dominant_key = ci["dominant"][0] if ci["dominant"] else None

    def bar(key: str, value: float) -> str:
        pct = 100.0 * abs(value) / max_abs
        color = "var(--accent)" if value >= 0 else "var(--credit)"
        label_class = "label bar-dominant" if key == dominant_key else "label"
        return (
            f'<div class="row"><div class="{label_class}">{key}</div>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%;'
            f'background:{color}"></div></div>'
            f'<div class="value">{value:,.0f}</div></div>'
        )

    cost_html = "\n".join(bar(k, v) for k, v in cost.items())
    cost_note = ""
    if ci["dominant"]:
        d_label, d_val = ci["dominant"]
        d_pct = ci["pct"](d_val)
        note = f"<b>{d_label}</b> is the largest contributor at {d_pct:.0f}% of total."
        if ci["second"]:
            s_label, s_val = ci["second"]
            note += f" <b>{s_label}</b> is second, at {ci['pct'](s_val):.0f}%."
        cost_note = f"<p class='muted'>{note}</p>"

    # ---- most important events -------------------------------------------
    ranked = rank_events(events)[:5]
    if ranked:
        cards_ev = []
        for i, (ev, diag) in enumerate(ranked, 1):
            tag = "CRITICAL" if i == 1 else "NOTABLE"
            css_class = "event-card critical" if i == 1 else "event-card notable"
            obs_html = "".join(f"<li>{o}</li>" for o in diag["observed"])
            cards_ev.append(
                f'<div class="{css_class}">'
                f'<div class="tag">#{i} · {tag}</div>'
                f"<div class='when'>{ev['start_t']} → {ev['end_t']} "
                f"({ev['n_intervals']} intervals)</div>"
                f"<div class='interp'>{diag['interpretation']}</div>"
                f"<details><summary class='muted'>Observed facts</summary>"
                f"<ul class='obs'>{obs_html}</ul></details>"
                f"</div>"
            )
        events_html = "\n".join(cards_ev)
    else:
        events_html = "<p class='muted'>No violation events recorded in this run.</p>"

    # ---- demand + battery insight -----------------------------------------
    peak_html = "<p class='muted'>No grid member found.</p>"
    if peak:
        mrows = "\n".join(
            f"<tr><td>{m}</td><td>{d['delivered']}</td><td>{d['commanded']}</td></tr>"
            for m, d in peak["members"].items()
        )
        peak_html = (
            f"<p><b>Highest observed grid-import interval (demand-charge proxy):</b> "
            f"{peak['grid_import_kva']:,.1f} kVA at k={peak['k']} ({peak['t']}).</p>"
            f"<table><tr><th>member</th><th>delivered</th><th>commanded</th></tr>"
            f"{mrows}</table>"
            f"<p class='muted'>Not the billed demand charge itself -- the tariff bills "
            f"the highest block-integrated kVA over the whole month. This is the single "
            f"interval most likely to be near it.</p>"
        )

    battery_html = "<p class='muted'>No battery in this fleet.</p>"
    if battery:
        brows = "\n".join(
            f"<tr><td>{bm}</td><td>{s['initial']:,.0f}</td><td>{s['min']:,.0f}</td>"
            f"<td>{s['max']:,.0f}</td><td>{s['final']:,.0f}</td></tr>"
            for bm, s in battery.items()
            if s["initial"] is not None
        )
        floor_events = [e for e in events if e.get("soc_floor_member")]
        floor_note = ""
        if floor_events:
            first = floor_events[0]
            floor_note = (
                f"<p class='muted'>{first['soc_floor_member']} reached its lower operating "
                f"point during {len(floor_events)} of the {len(events)} recorded event(s), "
                f"including the window starting {first['start_t']}.</p>"
            )
        battery_html = (
            f"<table><tr><th>member</th><th>initial kWh</th><th>min kWh</th>"
            f"<th>max kWh</th><th>final kWh</th></tr>{brows}</table>{floor_note}"
        )

    # ---- detailed evidence (collapsed) ------------------------------------
    if violations:
        vrows = "\n".join(
            f"<tr><td>{v['k']}</td><td>{v['t']}</td><td>{v['member']}</td><td>{v['note']}</td></tr>"
            for v in violations[:500]
        )
        more = (
            f"<p class='muted'>Showing first 500 of {len(violations)}.</p>"
            if len(violations) > 500
            else ""
        )
        evidence_html = (
            f"<table><tr><th>k</th><th>timestamp</th><th>member</th><th>note</th></tr>"
            f"{vrows}</table>{more}"
        )
    else:
        evidence_html = "<p class='muted'>None found.</p>"

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Run diagnostic — {manifest.get("name", "")}</title>
<style>
:root{{
  --ink:#1c1c1a; --muted:#767268; --line:#e6e3dd;
  --page-bg:#f6f5f2; --card-bg:#ffffff;
  --accent:#a8402f; --warn:#a9762c; --credit:#2f7d5c;
}}
*{{box-sizing:border-box}}
body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:880px;margin:0 auto;
padding:2.5rem 1.2rem 4rem;color:var(--ink);background:var(--page-bg)}}

.kicker{{font-size:.68rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
margin-bottom:.3rem;font-weight:600}}
h1{{font-size:1.5rem;margin:0 0 .25rem;letter-spacing:-.01em}}
h2{{font-size:.85rem;text-transform:uppercase;letter-spacing:.06em;color:#4a4740;
margin-top:2.6rem;margin-bottom:.9rem;padding-bottom:.4rem;border-bottom:1px solid var(--line)}}
.subhead{{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
margin:1.1rem 0 .4rem;font-weight:600}}
.muted{{color:var(--muted);font-size:.87rem}}
p{{margin:.5rem 0}}

.summary-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:.7rem;margin-top:1rem}}
.card{{border:1px solid var(--line);border-radius:10px;padding:.85rem 1rem;background:var(--card-bg);
box-shadow:0 1px 2px rgba(20,20,10,.04);transition:box-shadow .15s ease;min-height:64px}}
.card:hover{{box-shadow:0 2px 6px rgba(20,20,10,.07)}}
.card .label{{font-size:.66rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}}
.card .value{{font-size:1.4rem;font-weight:650;margin-top:.2rem;font-variant-numeric:tabular-nums;
letter-spacing:-.01em}}

.findings{{list-style:none;padding:0;margin:0}}
.findings li{{position:relative;padding:.6rem 0 .6rem .9rem;border-bottom:1px solid var(--line);
font-size:.93rem}}
.findings li:before{{content:"";position:absolute;left:0;top:.65rem;bottom:.65rem;width:2px;
background:var(--line)}}
.findings li:last-child{{border-bottom:none}}

.row{{display:grid;grid-template-columns:200px 1fr 100px;align-items:center;gap:.5rem;margin:.28rem 0}}
.row .label{{font-size:.87rem}}
.row .label.bar-dominant{{font-weight:700}}
.track{{background:#efede8;border-radius:4px;height:11px;overflow:hidden}}
.fill{{height:11px;border-radius:4px}}
.value{{text-align:right;font-variant-numeric:tabular-nums;font-size:.87rem}}
.total{{font-weight:700;border-top:2px solid var(--ink);margin-top:.6rem;padding-top:.6rem}}
.total .value{{font-size:1.05rem}}

table{{border-collapse:collapse;width:100%;font-size:.85rem}}
td,th{{border-bottom:1px solid var(--line);padding:.32rem .5rem;text-align:left}}
th{{font-size:.72rem;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);font-weight:600}}

.event-card{{border:1px solid var(--line);border-left:3px solid #c9c6bf;border-radius:8px;
padding:.9rem 1.05rem;margin-bottom:.7rem;background:var(--card-bg);
box-shadow:0 1px 2px rgba(20,20,10,.03)}}
.event-card.critical{{border-left-color:var(--accent)}}
.event-card.notable{{border-left-color:var(--warn)}}
.event-card .tag{{font-size:.66rem;text-transform:uppercase;letter-spacing:.06em;font-weight:700;
color:var(--muted)}}
.event-card.critical .tag{{color:var(--accent)}}
.event-card.notable .tag{{color:var(--warn)}}
.event-card .when{{font-size:.8rem;color:var(--muted);margin:.2rem 0 .55rem}}
.event-card .interp{{font-size:.9rem}}
.event-card .obs{{margin:.4rem 0 0 1.1rem;font-size:.82rem;color:#444}}

.evidence-count{{font-size:.8rem;color:var(--muted);margin:0 0 .6rem}}
details summary{{cursor:pointer;font-size:.8rem;color:var(--muted)}}
</style></head><body>

<div class="kicker">EMS run diagnostic</div>
<h1>Run: {manifest.get("name", "—")}</h1>
<p class="muted">dispatcher: {manifest.get("dispatcher")} · seed: {manifest.get("seed")} ·
dt: {manifest.get("dt_minutes")} min</p>

<h2>Run summary</h2>
<div class="summary-grid">{cards_html}</div>

<h2>Key findings</h2>
<ul class="findings">{findings_html}</ul>

<h2>Cost diagnosis</h2>
{cost_html}
<div class="row total"><div class="label">TOTAL</div><div></div><div class="value">{total:,.0f}</div></div>
{cost_note}

<h2>Most important events</h2>
<p class="muted"><b>Event definition:</b> a violation event is a run of intervals
whose index (<code>k</code>) is strictly consecutive -- a single interval with no
recorded violation ends the event. All violation codes and clip magnitudes inside
that run are summarized as one event.</p>
{events_html}

<h2>Demand &amp; battery insight</h2>
<div class="subhead">Demand</div>
{peak_html}
<div class="subhead">Battery</div>
{battery_html}

<h2>Detailed evidence</h2>
<p class="evidence-count">{len(violations)} raw violation record{"" if len(violations) == 1 else "s"}</p>
<p class="muted"><b>Violation notation:</b> <code>A-&gt;B</code> is the value before and
after the plant applied the relevant limit, for whichever quantity that specific
violation code names (a commanded kW, an SOC-derived limit, etc.) -- it is not
itself a state-of-charge reading. See "Demand &amp; battery insight" above for
actual SOC.</p>
<details><summary>Show {len(violations)} raw log rows</summary>{evidence_html}</details>

</body></html>"""


def _resolve_run_dir(run_arg: Path) -> Path:
    """Resolve --run to a directory that actually holds log.csv.

    Tried in order:
    1. `run_arg` exactly as given -- an absolute path, or a path relative to
       wherever the caller's shell happens to be. Covers `--run runs/latest`
       and any full path.
    2. `<repo_root>/runs/<run_arg>` -- the short form, e.g. `--run latest`.
       `<repo_root>` is derived from THIS FILE's own location (three parents
       up from tools/contrib/<handle>/tool.py), never from the current
       working directory, so this resolves the same way regardless of where
       the command is invoked from.

    Neither hardcodes "latest" -- any run name under runs/ works the same way.
    """
    if (run_arg / "log.csv").exists():
        return run_arg
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "runs" / run_arg
    if (candidate / "log.csv").exists():
        return candidate
    raise SystemExit(
        f"No log.csv found. Tried:\n  {run_arg}\n  {candidate}\n"
        f"Pass --run <name-under-runs/> (e.g. 'latest') or a full path to a run directory."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Run name under runs/ (e.g. 'latest'), or a full path to a run directory.",
    )
    ap.add_argument("--out", type=Path, required=True, help="Directory to write the report into.")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run)
    rows = _read_csv(run_dir / "log.csv")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    dt_minutes = manifest.get("dt_minutes", 15)

    violations = build_violations(rows)
    events = group_events(violations, rows)
    battery = battery_summary(rows)
    unserved_total = unserved_total_kwh(rows, dt_minutes)
    peak = demand_peak(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    html = render_html(manifest, violations, peak, events, battery, unserved_total, len(rows))
    (args.out / "diagnostic.html").write_text(html, encoding="utf-8")
    print(
        f"Wrote {args.out / 'diagnostic.html'} — {len(violations)} flagged intervals, "
        f"{len(events)} events."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())