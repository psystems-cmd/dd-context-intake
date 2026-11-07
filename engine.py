#!/usr/bin/env python3
"""DD Context Engine: merge role-weighted claims into one picture (pure functions, stdlib only).
  python3 engine.py <examples dir> [--bearers ops-1,ceo-1] [--out dir]
reads <dir>/claims/*.json, writes target_inventory.csv, INDEX.md, tensions.md.
The model (weights.yaml, schemas/dd-modules.json) is read from the model repo, never copied."""
import csv, io, json, os, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_FIELDS = ["name", "vendor", "category", "users", "cost_per_year", "last_used", "owner_role", "holds_data",
              "notes", "confidence", "sources"]
TARGET_ROLES = ["ceo", "it_ops_lead", "field_lead", "finance_lead"]
SHORT = {"ceo": "ceo", "it_ops_lead": "ops", "field_lead": "field", "finance_lead": "finance",
         "ucm_deal_lead": "deal", "upload": "upload"}
VERIFY = {  # per-section default "how to verify"; a claim's own `verify` field wins
    ("DATA", "data_quality"): "Export the student list (Excel) and count rows without ID and duplicate emails.",
    ("PROC", "timesheet_to_payout"): "Ask for the last payout run: Clockify export, payroll upload file, time spent checking.",
    ("COST", "margin_by_placement_type"): "Finance data: revenue and payroll cost per placement type, last 12 months (sevDesk + DATEV export).",
}
HINTS = {
    "team_structure": "How is the team organised: roughly how many people, which teams, which cities?",
    "decision_rights": "Who decides what: tools, clients, pricing, hiring (by role)?",
    "constraints_and_non_negotiables": "What are the hard constraints: contracts, deadlines, legal rules?",
    "what_must_not_break": "What must keep running from Day 1, no matter what?",
    "system_list": "Which tools do you use day to day? For each: what for, who owns it (role), how many users, last used.",
    "dependencies": "Which systems feed each other, automatically or by hand?",
    "contract_and_notice": "Which tool contracts run annually, and are there notice periods you know of?",
    "record_types": "Which kinds of records do you keep (students, companies, jobs...) and where?",
    "field_names": "What fields does the main student record have (no actual values, just the column names)?",
    "volumes": "Roughly how many records are there, and how many are active?",
    "unique_ids": "Does each record have a unique ID? What is used as the key?",
    "data_quality": "How clean is the data: duplicates, gaps, outdated rows?",
    "where_documents_live": "Where do documents live: contracts, certificates, templates, process notes?",
    "duplicate_and_outdated_versions": "Are there duplicate or outdated versions people still use?",
    "owners": "Who (by role) owns which document collections?",
    "placement": "Walk me through one placement: steps, systems, manual work, time per placement.",
    "onboarding": "How is a new student onboarded: steps, systems, manual work, time per student?",
    "timesheet_to_payout": "From timesheet to payout: steps, systems, manual work, time per payout cycle?",
    "billing": "How are clients billed: steps, systems, manual work, time per billing run?",
    "knowledge_holders": "Which role knows which system or process best (role and topic, no judgement)?",
    "ai_tools_in_use": "Which AI tools are in use, officially or not?",
    "accounts": "Company accounts or private accounts?",
    "homemade_agents": "Has anyone built their own GPTs, agents or automations with AI?",
    "data_put_into_them": "What kind of data goes into these AI tools?",
    "licence_costs": "Roughly what do licences cost per year (ranges only before closing)?",
    "contract_terms": "Which contracts matter most and what are their terms?",
    "notice_periods": "What notice periods apply to the main contracts?",
    "margin_by_placement_type": "What is the margin per placement type (ranges only before closing)?",
}


# ---------- model ----------
def _scalar(v):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        return [_scalar(x) for x in v[1:-1].split(",") if x.strip()]
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v

def parse_weights(text):
    """Exactly the weights.yaml shape: top-level scalars, inline [lists], and `roles:` with one `key: {map}` per line."""
    out, block = {}, None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        key, _, val = line.strip().partition(":")
        val = val.strip()
        if raw[0] in " \t" and block:
            out[block][key] = {k.strip(): _scalar(x) for k, _, x in
                               (p.partition(":") for p in val.strip("{}").split(","))}
        elif not val:
            block, out[key] = key, {}
        else:
            block, out[key] = None, _scalar(val)
    return out

def load_model(model_dir=None):
    d = Path(os.environ.get("DD_MODEL_DIR") or model_dir or ROOT / "../structured-context-model/dd")
    if not d.is_absolute():
        d = ROOT / d
    weights = parse_weights((d / "weights.yaml").read_text())
    modules = json.loads((d / "schemas" / "dd-modules.json").read_text())["modules"]
    return {"dir": str(d.resolve()), "weights": weights, "modules": modules, "mods": {m["id"]: m for m in modules}}

def weight(model, role, module):
    if role == "upload":
        return float(model["weights"].get("data_weight", 1.0))
    return float(model["weights"]["roles"].get(role, {}).get(module, 0))

def role_label(model, role):
    return "Uploaded file (data)" if role == "upload" else model["weights"]["roles"].get(role, {}).get("label", role)


# ---------- merge ----------
def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-9)

def disagree(a, b, pct=20):
    """Tension rules: number >pct apart; number vs range: an endpoint >pct off the number; state differs;
    list: an item one claim lists is denied by the other. No norm on either side: no tension."""
    na, nb, t = a.get("norm") or {}, b.get("norm") or {}, pct / 100
    if not na or not nb:
        return False
    if "state" in na and "state" in nb:
        return na["state"] != nb["state"]
    if ("items" in na or "denies" in na) and ("items" in nb or "denies" in nb):
        low = lambda n, k: {str(x).lower() for x in n.get(k, [])}
        return bool(low(na, "denies") & low(nb, "items") or low(nb, "denies") & low(na, "items"))
    if "number" in na and "number" in nb:
        return _rel(na["number"], nb["number"]) > t
    if "range" in na and "range" in nb:
        return any(_rel(x, y) > t for x, y in zip(na["range"], nb["range"]))
    num, rng = (na, nb) if "number" in na else (nb, na)
    if "number" in num and "range" in rng:
        n = num["number"]
        return any(abs(e - n) / max(abs(n), 1e-9) > t for e in rng["range"])
    return False

def has_content(c, sec, model, module):
    if c.get("not_applicable"):
        return weight(model, c["source_role"], module) >= model["weights"].get("deep_min", 0.7)
    req = sec.get("requires")
    if req:
        f = c.get("fields") or {}
        return all(str(f.get(k, "")).strip() for k in req)
    return bool(str(c.get("value", "")).strip())

def _rank(model, module):
    return lambda c: (c.get("kind") == "data", weight(model, c["source_role"], module), c.get("timestamp", ""))

def brief(c, model, module):
    b = {k: c.get(k) for k in ("bearer_id", "source_role", "source_id", "kind", "value", "quote")}
    b.update(weight=weight(model, c["source_role"], module), role_label=role_label(model, c["source_role"]))
    return b

def merge(claims, model):
    pct = model["weights"].get("number_disagreement_pct", 20)
    by = defaultdict(list)
    for c in claims:
        by[(c["module"], c["section"])].append(c)
    out, tensions = {}, []
    for mod in model["modules"]:
        m = mod["id"]
        secs, n_t, roles = [], 0, set()
        for sec in mod["sections"]:
            cs = sorted(by[(m, sec["id"])], key=_rank(model, m), reverse=True)
            roles |= {c["source_role"] for c in cs}
            is_data = "data" in sec["type"]
            content = [c for c in cs if has_content(c, sec, model, m)]
            ts = []
            if cs:
                top = cs[0]
                for o in cs[1:]:
                    if disagree(top, o, pct):
                        ts.append({"module": m, "section": sec["id"], "winner": brief(top, model, m),
                                   "other": brief(o, model, m), "_claims": (top, o),
                                   "verify": o.get("verify") or top.get("verify") or VERIFY.get(
                                       (m, sec["id"]), "Ask for the export or document that settles it.")})
            n_t += len(ts)
            tensions += ts
            secs.append({"id": sec["id"], "essential": sec["essential"], "type": sec["type"], "claims": len(cs),
                         "covered": bool(content), "tensions": len(ts),
                         "working": brief(cs[0], model, m) if cs else None,
                         "needs_verification": bool(cs) and (bool(ts) or (is_data and cs[0].get("kind") != "data")),
                         "grounding": max([1.0 if c.get("kind") == "data" else 0.5 for c in content] or [0.0])})
        ess = [s for s in secs if s["essential"]]
        cov = sum(s["covered"] for s in ess) / len(ess)
        target_w = sum(weight(model, r, m) for r in TARGET_ROLES)
        div = sum(weight(model, r, m) for r in roles if r in TARGET_ROLES) / target_w if target_w else 0
        data_secs = [s for s in secs if "data" in s["type"]]
        ground = sum(s["grounding"] for s in data_secs) / len(data_secs) if data_secs else cov
        cons = max(0.5, 1 - 0.1 * n_t)
        conf = 0.35 * cov + 0.25 * div + 0.20 * ground + 0.10 * cons + 0.10 * 1.0 if roles else 0.0
        out[m] = {"id": m, "name": mod["name"], "feeds": mod.get("feeds", ""), "confidence": round(conf, 2),
                  "essential_coverage": [sum(s["covered"] for s in ess), len(ess)],
                  "sources": sorted(({"role": r, "label": role_label(model, r), "weight": weight(model, r, m)}
                                     for r in roles), key=lambda s: -s["weight"]),
                  "tensions": n_t, "needs_verification": [s["id"] for s in secs if s["needs_verification"]],
                  "missing": [s["id"] for s in ess if not s["covered"]], "sections": secs}
    contributors = sorted({c["bearer_id"] for c in claims if c["source_role"] != "upload"})
    return {"modules": out, "tensions": tensions, "contributors": contributors,
            "next_role": next_role(out, {c["source_role"] for c in claims}, model), "claims": len(claims)}

def next_role(mods, contributing, model):
    """The role with the highest weight in the two thinnest modules, preferring roles not yet heard."""
    thin = sorted(mods.values(), key=lambda m: m["confidence"])[:2]
    cands = [r for r in TARGET_ROLES if r not in contributing] or TARGET_ROLES
    best = max(cands, key=lambda r: sum(weight(model, r, m["id"]) for m in thin))
    return {"role": best, "label": role_label(model, best), "modules": [m["id"] for m in thin],
            "new": best not in contributing}


# ---------- exports ----------
def _short(roles):
    return "+".join(dict.fromkeys(SHORT.get(r, r) for r in roles))

NOTE_SECTIONS = {"DATA": {"unique_ids", "data_quality"}, "AI": {"accounts", "homemade_agents", "data_put_into_them"}}

def inventory_rows(claims, merged, model):
    """One row per system: SYS.system_list claims + uploaded rows (top-ranked claim fills fields first),
    notes enriched from DATA/AI claims naming the system, with tension markers."""
    sys_claims = [c for c in claims if c["module"] == "SYS" and c["section"] == "system_list"
                  and (c.get("fields") or {}).get("name")]
    rows = {}  # insertion order = first mention
    for c in sys_claims:
        rows.setdefault(c["fields"]["name"].strip().lower(), []).append(c)
    losers = {id(t["_claims"][1]) for t in merged["tensions"]}
    winners = {id(t["_claims"][0]) for t in merged["tensions"]}
    best = {}  # top-ranked claim per DATA/AI section, as merge picked it
    for mod in NOTE_SECTIONS:
        for c in sorted([c for c in claims if c["module"] == mod], key=_rank(model, mod), reverse=True):
            best.setdefault((mod, c["section"]), c)
    out = []
    for key, cs in rows.items():
        cs = sorted(cs, key=_rank(model, "SYS"), reverse=True)
        row, roles, notes, tension = {k: "" for k in CSV_FIELDS}, [c["source_role"] for c in cs], [], False
        for c in cs:
            for k in CSV_FIELDS[:8]:
                row[k] = row[k] or str(c["fields"].get(k, "")).strip()
            n = c["fields"].get("notes") or c["fields"].get("purpose")
            if n and n not in notes:
                notes.append(n)
        for c in claims:
            if c["section"] not in NOTE_SECTIONS.get(c["module"], ()) or \
                    key not in {s.strip().lower() for s in c.get("systems_mentioned", [])}:
                continue
            roles.append(c["source_role"])
            if id(c) in losers:
                notes.append(f"{role_label(model, c['source_role'])} says {c['value']} (tension)")
                tension = True
            elif best.get((c["module"], c["section"])) is c and c["value"] not in notes:
                notes.append(c["value"])
                tension |= id(c) in winners
        base = weight(model, cs[0]["source_role"], "SYS") * (1.0 if cs[0].get("kind") == "data" else 0.8)
        conf = min(1.0, max(0.0, base + 0.05 * (len(set(roles)) - 1) - (0.15 if tension else 0)))
        row.update(notes="; ".join(notes), confidence=f"{conf:.2f}", sources=_short(roles))
        out.append(row)
    return out

def render_csv(rows):
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    wr.writeheader()
    wr.writerows(rows)
    return buf.getvalue()

def render_index(merged, model, target):
    L = [f"# INDEX: {target} (test data)", "",
         f"Merged from {merged['claims']} claims by {len(merged['contributors'])} contributors "
         f"({', '.join(merged['contributors'])}). Roles only, no names.", ""]
    for m in merged["modules"].values():
        ec = m["essential_coverage"]
        L += [f"## {m['id']} · {m['name']}", "",
              f"- confidence: {m['confidence']:.2f}",
              f"- essential_coverage: {ec[0]}/{ec[1]}",
              "- sources: " + (", ".join("%s (%s)" % (s["role"], s["weight"]) for s in m["sources"]) or "none"),
              f"- tensions: {m['tensions']}",
              f"- needs_verification: {len(m['needs_verification'])}" +
              (f" ({', '.join(m['needs_verification'])})" if m["needs_verification"] else ""),
              f"- missing: {', '.join(m['missing']) or 'none'}", f"- feeds: {m['feeds']}"]
        if m["id"] == "SYS":
            ok = m["confidence"] >= 0.5
            L.append(f"- decision rule: SYS {m['confidence']:.2f} {'>=' if ok else '<'} 0.50 → " +
                     ("classifier may produce a docking plan" if ok else
                      "classifier refuses the docking plan; missing: " + (", ".join(m["missing"]) or "sources")))
        L.append("")
    nr = merged["next_role"]
    L.append(f"**Next interview:** {nr['label']} ({nr['role']}), highest weight in the thinnest modules "
             f"({', '.join(nr['modules'])}){'' if nr['new'] else '; role already heard, add a second person'}.")
    return "\n".join(L) + "\n"

def render_tensions(merged, target):
    L = [f"# Tensions: {target} (test data)", "", f"{len(merged['tensions'])} tensions. "
         "Both claims are kept; the higher-ranked one is the working value until verified.", ""]
    for i, t in enumerate(merged["tensions"], 1):
        L += [f"## {i}. {t['module']} · {t['section']}", ""]
        for c in (t["winner"], t["other"]):
            L.append(f"- **{c['source_role']}** ({c['bearer_id']}, weight {c['weight']}, {c['kind']}): "
                     f"{c['value']} — \"{c['quote']}\"")
        w = t["winner"]
        L += ["", f"- suggested winner: {w['source_role']} ({w['bearer_id']}): "
              + ("data beats estimate" if w["kind"] == "data" and t["other"]["kind"] != "data"
                 else f"higher weight for {t['module']} ({w['weight']} vs {t['other']['weight']})"),
              f"- how to verify: {t['verify']}", ""]
    return "\n".join(L)

def export(claims, model, target):
    merged = merge(claims, model)
    return {"target_inventory.csv": render_csv(inventory_rows(claims, merged, model)),
            "INDEX.md": render_index(merged, model, target), "tensions.md": render_tensions(merged, target)}

def coverage(claims, model):
    """Coverage by content per module (for the interview sidebar)."""
    out = {}
    for mod in model["modules"]:
        ess = [s for s in mod["sections"] if s["essential"]]
        done = [s["id"] for s in ess if any(c["module"] == mod["id"] and c["section"] == s["id"]
                                            and has_content(c, s, model, mod["id"]) for c in claims)]
        out[mod["id"]] = {"covered": done, "total": len(ess)}
    return out

def questions(model, role):
    """Scripted stand-in for the adaptive interviewer: deep modules first (all essential sections),
    then light modules (first two essential sections); modules below light_min are skipped."""
    W = model["weights"]
    mods = sorted((m for m in model["modules"] if weight(model, role, m["id"]) >= W.get("light_min", 0.4)),
                  key=lambda m: -weight(model, role, m["id"]))
    qs = []
    for m in mods:
        deep = weight(model, role, m["id"]) >= W.get("deep_min", 0.7)
        secs = [s for s in m["sections"] if s["essential"]][: None if deep else 2]
        qs += [{"module": m["id"], "section": s["id"], "depth": "deep" if deep else "light",
                "requires": s.get("requires", []),
                "text": HINTS.get(s["id"], f"{m['name']}: tell us about {s['id'].replace('_', ' ')}.")} for s in secs]
    return qs

def load_claims(examples_dir, bearers=None):
    claims = []
    for p in sorted(Path(examples_dir, "claims").glob("*.json")):
        claims += json.loads(p.read_text())
    return [c for c in claims if not bearers or c["bearer_id"] in bearers]


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        sys.exit(__doc__)
    opt = dict(zip(args[1::2], args[2::2]))
    ex = Path(args[0])
    claims = load_claims(ex, opt.get("--bearers", "").split(",") if "--bearers" in opt else None)
    model = load_model(json.loads((ROOT / "config.json").read_text()).get("model_dir") if (ROOT / "config.json").exists() else None)
    out = Path(opt.get("--out", ex / "expected"))
    out.mkdir(parents=True, exist_ok=True)
    target = claims[0]["target"] if claims else ex.name
    for name, text in export(claims, model, target).items():
        (out / name).write_text(text)
    m = merge(claims, model)
    print(f"{len(claims)} claims, {len(m['contributors'])} contributors, {len(m['tensions'])} tensions → {out}")
    for x in m["modules"].values():
        print(f"  {x['id']:<5} conf {x['confidence']:.2f}  coverage {x['essential_coverage'][0]}/{x['essential_coverage'][1]}  "
              f"tensions {x['tensions']}  sources {_short(s['role'] for s in x['sources'])}")
    print(f"  next role: {m['next_role']['role']}")
