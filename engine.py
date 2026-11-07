#!/usr/bin/env python3
"""DD Context Engine: merge what several people at a target said into one picture (pure functions, stdlib only).
  python3 engine.py <example dir> [--out dir]
reads <dir>/people.json + <dir>/interviews/*.json, writes target_inventory.csv, INDEX.md, tensions.md (default: <dir>/expected).
The model (weights.yaml, schemas/dd-modules.json) is read from the model repo, never copied.

Data shapes (see README):
  target  = {slug, name, people: [person], answers: [answer], upload?: {file, rows, claims}}
  person  = {id, first, last, email, role, note, token, status}       names live ONLY here
  answer  = {person_id, module, section, question, answer, not_applicable, at, claims: [claim]}
  claim   = {bearer_id, role, module?, section?, value, kind: estimate|data, norm?, fields?, system?, systems?, topic?, verify?}
            module/section default to the answer's; extraction may map a statement to another section."""
import csv, io, json, os, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_FIELDS = ["name", "vendor", "category", "users", "cost_per_year", "last_used", "owner_role", "holds_data",
              "notes", "confidence", "sources"]
TARGET_ROLES = ["ceo", "ops_lead", "finance_lead", "team_member"]  # people at the target; ucm_deal_lead is internal
SHORT = {"ceo": "ceo", "ops_lead": "ops", "finance_lead": "finance", "team_member": "team", "ucm_deal_lead": "ucm",
         "upload": "file"}
ID_PREFIX = {"ceo": "ceo", "ops_lead": "ops", "finance_lead": "fin", "team_member": "team", "ucm_deal_lead": "ucm"}
STATUSES = ["not invited", "invited", "in progress", "done"]
VERIFY = {  # default "how to settle it" per section; a claim's own `verify` wins
    ("DATA", "record_quality"): "Export the student list and count rows without an ID and duplicate email addresses.",
    ("FLOW", "timesheet_to_payout"): "Ask for the last payout run: Clockify export, payroll upload file and the time spent checking it.",
    ("SYS", "tool_list"): "Ask for the tool's usage data (admin analytics or a last-edited export).",
}
DECIDES = {  # what the conflict decides in the docking plan
    ("SYS", "tool_list"): "whether the tool is switched off (K3 sunset) or kept",
    ("DATA", "record_quality"): "how much clean-up the records need before they move",
    ("FLOW", "timesheet_to_payout"): "whether payouts are a Day 1 risk (K1)",
}
REASON = {"SYS": "they run the tools day to day", "DATA": "they work in the records daily",
          "FLOW": "they do this work every day", "COST": "they handle the contracts and invoices",
          "RISK": "they keep the operation running"}


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
    """Exactly the weights.yaml shape: top-level scalars, inline [lists], and `roles:` with one
    `key: {label: .., max: n, SYS: 0.9, ..}` per line (max = people per target, 0 = any number)."""
    out, block = {}, None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        key, _, val = line.strip().partition(":")
        val = val.strip()
        if raw[0] in " \t" and block:
            m = {k.strip(): _scalar(x) for k, _, x in (p.partition(":") for p in val.strip("{}").split(","))}
            m["max"] = int(m.get("max", 0) or 0)
            m["label"] = str(m.get("label", key))
            out[block][key] = m
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
    return {"dir": str(d.resolve()), "weights": weights, "roles": weights["roles"], "modules": modules,
            "mods": {m["id"]: m for m in modules},
            "secs": {(m["id"], s["id"]): s for m in modules for s in m["sections"]}}

def weight(model, role, module):
    if role == "upload":
        return float(model["weights"].get("data_weight", 1.0))
    return float(model["roles"].get(role, {}).get(module, 0))

def label(model, role):
    return "Uploaded file" if role == "upload" else model["roles"].get(role, {}).get("label", role)

def sec_label(section):
    return section.replace("_", " ")

def questions(model, role):
    """The interview for a role: deep modules (weight >= deep_min) ask every section, light modules
    (>= light_min) ask the essential ones; below light_min the module is not this role's area."""
    W = model["weights"]
    mods = sorted((m for m in model["modules"] if weight(model, role, m["id"]) >= W.get("light_min", 0.4)),
                  key=lambda m: -weight(model, role, m["id"]))  # stable: ties keep schema order
    qs = []
    for m in mods:
        deep = weight(model, role, m["id"]) >= W.get("deep_min", 0.7)
        qs += [{"module": m["id"], "module_name": m["name"], "section": s["id"], "question": s["question"],
                "depth": "deep" if deep else "light"} for s in m["sections"] if deep or s["essential"]]
    return qs


# ---------- people ----------
def new_id(role, people):
    p, taken = ID_PREFIX.get(role, role), {x["id"] for x in people}
    return next(f"{p}-{n}" for n in range(1, 999) if f"{p}-{n}" not in taken)

def check_person(model, people, person):
    """Raise ValueError if the role is unknown or already filled (role `max` from weights.yaml)."""
    role = person.get("role")
    if role not in model["roles"]:
        raise ValueError(f"Unknown role '{role}'.")
    mx = model["roles"][role]["max"]
    others = [p for p in people if p["role"] == role and p["id"] != person.get("id")]
    if mx and len(others) >= mx:
        who = " ".join(x for x in (others[0].get("first"), others[0].get("last")) if x) or others[0]["id"]
        raise ValueError(f"There can only be {'one' if mx == 1 else mx} {label(model, role)} per target, and {who} "
                         f"already has this role. Edit that person, or pick another role (any number of team members is fine).")
    for k in ("first", "last", "email"):
        if not str(person.get(k, "")).strip():
            raise ValueError("First name, last name and email are required (the email can be a dummy).")


# ---------- claims ----------
def claims_of(target):
    """Flatten answers (+ an uploaded tool list) into claims. Each claim keeps bearer_id + role, never a name."""
    out = []
    for i, a in enumerate(target.get("answers", [])):
        for c in a.get("claims", []):
            out.append({"module": a["module"], "section": a["section"], "kind": "estimate", **c,
                        "at": a.get("at", ""), "ref": i})
    for c in (target.get("upload") or {}).get("claims", []):
        out.append({"module": "SYS", "section": "tool_list", "kind": "data", "bearer_id": "upload", "role": "upload",
                    **c, "at": c.get("at", ""), "ref": None})
    return out

def stub_claim(person, module, section, text, not_applicable=False):
    """Live interview stand-in for the extraction prompt: the answer becomes one text claim."""
    return {"bearer_id": person["id"], "role": person["role"], "module": module, "section": section,
            "value": "doesn't apply" if not_applicable else text, "kind": "estimate",
            "not_applicable": bool(not_applicable), "extraction": "stub"}

def _name(n):
    return " ".join(str(n or "").lower().split())

def mentions(c):
    """Systems a claim is about (normalised names)."""
    f = c.get("fields") or {}
    names = [f.get("name"), c.get("system"), *(c.get("systems") or []), *(f.get("systems") or [])]
    return {_name(n) for n in names if n}

def has_content(c, sec, model, module):
    if c.get("not_applicable"):
        return weight(model, c["role"], module) >= model["weights"].get("deep_min", 0.7)
    req = sec.get("requires")
    if req:
        f = c.get("fields") or {}
        return all(str(f.get(k, "")).strip() for k in req)
    return bool(str(c.get("value", "")).strip())

def _rank(model):
    """Merge rule: data beats estimate, then role weight for the module, then recency."""
    return lambda c: (c.get("kind") == "data", weight(model, c["role"], c["module"]), c.get("at", ""))


# ---------- merge ----------
def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-9)

def disagree(a, b, pct=20):
    """Tension rules (on `norm` only): state differs; number >pct apart; number vs range: an endpoint >pct off;
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

def _and(xs):
    xs = list(dict.fromkeys(xs))
    return " and ".join([", ".join(xs[:-1]), xs[-1]] if len(xs) > 1 else xs)

def conflicts(claims, model):
    """Compare normed claims per section (per system for SYS.tool_list). The top-ranked claim sets the working
    value; claims that disagree with it form the other side. The winning side's spokesperson is its most
    complete claim (so a detailed answer speaks for the people who agree with it)."""
    pct, rank = model["weights"].get("number_disagreement_pct", 20), _rank(model)
    order = {k: i for i, k in enumerate(model["secs"])}
    groups = defaultdict(list)
    for c in claims:
        if c.get("norm") and (c["module"], c["section"]) in order:
            sysname = (c.get("fields") or {}).get("name") if c["section"] == "tool_list" else None
            groups[(c["module"], c["section"], _name(sysname) if sysname else "")].append(c)
    out = []
    for key in sorted(groups, key=lambda k: (order[k[:2]], k[2])):
        m, s, _ = key
        cs = sorted(groups[key], key=rank, reverse=True)
        lose = [c for c in cs if disagree(cs[0], c, pct)]
        if not lose:
            continue
        win = [c for c in cs if not any(c is x for x in lose)]
        sec = model["secs"][(m, s)]
        w = max(win, key=lambda c: (c.get("kind") == "data", has_content(c, sec, model, m)) + rank(c))
        o = lose[0]
        system = ((w.get("fields") or {}).get("name") if s == "tool_list" else None) or w.get("system") or o.get("system")
        wl, ol = label(model, w["role"]), label(model, o["role"])
        agree = [label(model, c["role"]) for c in win if c is not w and c["role"] != w["role"]]
        reason = ("it is backed by a file or export" if w.get("kind") == "data" and o.get("kind") != "data"
                  else REASON.get(m, "they are the stronger source here"))
        verify = w.get("verify") or o.get("verify") or VERIFY.get((m, s), "Ask for the export or document that settles it.")
        topic = w.get("topic") or o.get("topic") or (f"{system}: {sec_label(s)}" if system else sec_label(s).capitalize())
        decides = DECIDES.get((m, s), "")
        sentence = (f"{wl} says “{w['value']}”" + (f" ({_and(agree)} {'agree' if len(set(agree)) > 1 else 'agrees'})" if agree else "")
                    + f". {ol} says “{o['value']}”. → We go with {wl} because {reason}. To settle it: {verify}")
        out.append({"id": f"c{len(out) + 1}", "module": m, "section": s, "topic": topic, "system": system,
                    "winner_role": wl, "winner_says": w["value"], "other_role": ol, "other_says": o["value"],
                    "verify": verify, "decides": decides, "reason": reason, "agree": list(dict.fromkeys(agree)),
                    "sentence": sentence,
                    "winner": {"role": w["role"], "bearer_id": w["bearer_id"], "kind": w.get("kind"),
                               "weight": weight(model, w["role"], m), "ref": w.get("ref")},
                    "other": {"role": o["role"], "bearer_id": o["bearer_id"], "kind": o.get("kind"),
                              "weight": weight(model, o["role"], m), "ref": o.get("ref")},
                    "refs": {"winner": [c.get("ref") for c in win], "other": [c.get("ref") for c in lose]},
                    "people": {"winner": list(dict.fromkeys(c["bearer_id"] for c in win)),
                               "other": list(dict.fromkeys(c["bearer_id"] for c in lose))}})
    return out


# ---------- coverage + who to ask next ----------
def role_states(people):
    """role -> 'done' (everyone with it finished) | 'pending' (someone not finished) ; absent roles are missing."""
    st = {}
    for p in people:
        st[p["role"]] = "pending" if p["status"] != "done" or st.get(p["role"]) == "pending" else "done"
    return st

def who_to_ask(model, people, module, section):
    """Never suggests interviewing a role that is already done: waiting for someone on the list whose interview asks
    this > add a role nobody has yet > follow-up question to someone already interviewed."""
    sec, st = model["secs"][(module, section)], role_states(people)
    asks = lambda r: any(q["module"] == module and q["section"] == section for q in questions(model, r))
    roles = sorted(model["roles"], key=lambda r: -weight(model, r, module))
    pending = [r for r in roles if st.get(r) == "pending" and asks(r)]
    if pending:
        r = pending[0]
        waiting = [p for p in people if p["role"] == r and p["status"] != "done"]
        invited = any(p["status"] != "not invited" for p in waiting)
        return {"kind": "waiting" if invited else "invite", "role": r, "label": label(model, r),
                "text": f"Waiting for the {label(model, r)} interview (their interview asks this)" if invited
                else f"Send the {label(model, r)} their interview link (their interview asks this)"}
    absent = [r for r in roles if r in TARGET_ROLES and r not in st and asks(r)]
    if absent:
        r = absent[0]
        return {"kind": "add", "role": r, "label": label(model, r), "text": f"Add a {label(model, r)} and invite them"}
    done = [r for r in roles if st.get(r) == "done" and weight(model, r, module) > 0]
    if done:
        r = done[0]
        return {"kind": "follow_up", "role": r, "label": label(model, r), "question": sec["question"],
                "text": f"Follow-up question to {label(model, r)} (already interviewed): “{sec['question']}”"}
    return {"kind": "none", "role": None, "label": None, "text": "Nobody on the list covers this yet: add people first"}

def coverage(claims, model, people=()):
    """Per module and section: covered by content (not by number of answers), who said something, who to ask next."""
    by = defaultdict(list)
    for c in claims:
        by[(c["module"], c["section"])].append(c)
    out = []
    for mod in model["modules"]:
        m, secs = mod["id"], []
        for s in mod["sections"]:
            cs = by[(m, s["id"])]
            ok = any(has_content(c, s, model, m) for c in cs)
            item = {"id": s["id"], "label": sec_label(s["id"]), "essential": s["essential"], "covered": ok,
                    "answered": bool(cs), "roles": list(dict.fromkeys(c["role"] for c in cs)),
                    "grounding": max([1.0 if c.get("kind") == "data" else 0.5 for c in cs if has_content(c, s, model, m)] or [0.0])}
            if not ok and cs and s.get("requires"):
                have = set().union(*[{k for k, v in (c.get("fields") or {}).items() if str(v).strip()} for c in cs])
                item["thin"] = [sec_label(k) for k in s["requires"] if k not in have]
            item["ask"] = {"kind": "complete", "text": "complete"} if ok else who_to_ask(model, list(people), m, s["id"])
            secs.append(item)
        ess = [s for s in secs if s["essential"]]
        out.append({"module": m, "name": mod["name"], "feeds": mod.get("feeds", ""), "sections": secs,
                    "covered": sum(s["covered"] for s in ess), "total": len(ess),
                    "missing": [s["id"] for s in ess if not s["covered"]]})
    return out

def confidence(cov, claims, confl, model):
    """Details only (never on the main screen): 0.35 coverage + 0.25 source diversity + 0.20 data grounding
    + 0.10 consistency + 0.10 recency."""
    out = {}
    for mc in cov:
        m = mc["module"]
        roles = {c["role"] for c in claims if c["module"] == m}
        tw = sum(weight(model, r, m) for r in TARGET_ROLES)
        div = sum(weight(model, r, m) for r in roles if r in TARGET_ROLES) / tw if tw else 0
        cov_ = mc["covered"] / mc["total"] if mc["total"] else 0
        data = [s for s in mc["sections"] if "data" in model["secs"][(m, s["id"])]["type"]]
        ground = sum(s["grounding"] for s in data) / len(data) if data else cov_
        n_t = sum(t["module"] == m for t in confl)
        conf = 0.35 * cov_ + 0.25 * div + 0.20 * ground + 0.10 * max(0.5, 1 - 0.1 * n_t) + 0.10 if roles else 0.0
        out[m] = {"confidence": round(conf, 2), "coverage": round(cov_, 2), "diversity": round(div, 2),
                  "grounding": round(ground, 2), "conflicts": n_t,
                  "sources": sorted(({"role": r, "label": label(model, r), "weight": weight(model, r, m)} for r in roles),
                                    key=lambda s: -s["weight"])}
    return out


# ---------- systems (the inventory) ----------
def systems(claims, confl, model):
    """One row per system named in SYS.tool_list (interviews + uploaded file). Fields come from the top-ranked
    claim, blanks filled from the others; cost prefers COST.tool_costs; notes add what other sections said about
    the system and a (conflict: ...) marker."""
    rank = _rank(model)
    tl = [c for c in claims if c["module"] == "SYS" and c["section"] == "tool_list" and (c.get("fields") or {}).get("name")]
    groups = {}
    for c in tl:
        groups.setdefault(_name(c["fields"]["name"]), []).append(c)
    by_conf = {_name(t["system"]): t for t in confl if t["system"]}
    losers = {id(c) for c in claims for t in confl if c.get("ref") in t["refs"]["other"] and c.get("norm")
              and c["module"] == t["module"] and c["section"] == t["section"]}
    rows = []
    for key, cs in groups.items():
        cs = sorted(cs, key=rank, reverse=True)
        f = {k: "" for k in CSV_FIELDS}
        for c in cs:
            for k in CSV_FIELDS[:8]:
                f[k] = f[k] or str(c["fields"].get(k, "")).strip()
        cost = sorted([c for c in claims if c["module"] == "COST" and c["section"] == "tool_costs"
                       and _name((c.get("fields") or {}).get("name")) == key and (c.get("fields") or {}).get("cost_per_year")],
                      key=rank, reverse=True)
        if cost:
            f["cost_per_year"] = str(cost[0]["fields"]["cost_per_year"])
        about = sorted([c for c in claims if key in mentions(c)], key=rank, reverse=True)
        roles = list(dict.fromkeys(c["role"] for c in sorted(about, key=rank, reverse=True)))
        purpose = cs[0]["fields"].get("purpose") or cs[0]["fields"].get("notes") or ""
        notes = [purpose] if purpose else []
        for c in about:
            if c["section"] not in ("tool_list", "tool_costs") and c.get("system") and _name(c["system"]) == key \
                    and id(c) not in losers and c["value"] not in notes:
                notes.append(c["value"])
        t = by_conf.get(key)
        if t:
            notes.append(f"(conflict: {t['other_role']} says {t['other_says']}; working value from {t['winner_role']}, verify: {t['verify']})")
        top = cs[0]
        conf = weight(model, top["role"], "SYS") * (1.0 if top.get("kind") == "data" else 0.8) \
            + 0.05 * (len(roles) - 1) - (0.15 if t else 0)
        f.update(notes="; ".join(notes), confidence=f"{min(1.0, max(0.0, conf)):.2f}",
                 sources=" + ".join(SHORT.get(r, r) for r in roles))
        rows.append({"row": f, "purpose": purpose, "roles": roles, "role_labels": [label(model, r) for r in roles],
                     "status": "conflict" if t else "agreed" if len(roles) > 1 else "only one source",
                     "conflict": t["id"] if t else None, "_w": -max(weight(model, c["role"], "SYS") for c in cs)})
    rows.sort(key=lambda r: r["_w"])  # stable: best source first, then order of first mention
    for r in rows:
        r.pop("_w")
    return rows


# ---------- results (everything the UI and exports need) ----------
def results(target, model):
    people, claims = target.get("people", []), claims_of(target)
    confl = conflicts(claims, model)
    cov = coverage(claims, model, people)
    syst = systems(claims, confl, model)
    done = [p for p in people if p["status"] == "done"]
    asks = [dict(s["ask"], module=c["module"], section=s["id"]) for c in cov for s in c["sections"]
            if s["essential"] and not s["covered"]]
    return {"systems": syst, "conflicts": confl, "coverage": cov, "details": confidence(cov, claims, confl, model),
            "interviews": {"done": len(done), "total": len(people),
                           "roles": [{"role": p["role"], "label": label(model, p["role"]), "status": p["status"]} for p in people]},
            "contributors": list(dict.fromkeys(c["bearer_id"] for c in claims if c["role"] != "upload")),
            "asks": asks, "next": next_step(target, asks, model), "step": step(target)}

def step(target):
    """Which of the 5 setup steps the deal lead is on: 1 add people, 2 send invites, 3 watch answers, 4 review, 5 send."""
    ppl = target.get("people", [])
    if not ppl:
        return 1
    if any(p["status"] == "not invited" for p in ppl):
        return 2
    if any(p["status"] != "done" for p in ppl):
        return 3
    return 5 if target.get("sent") else 4

def next_step(target, asks, model):
    ppl = target.get("people", [])
    if not ppl:
        return "Add the people at the target who will answer: start with the Head of Ops / IT and the CEO."
    n = sum(p["status"] == "not invited" for p in ppl)
    if n:
        return f"Create interview links for {n} {'person' if n == 1 else 'people'} (Send invites), then send each person their link."
    waiting = [p for p in ppl if p["status"] != "done"]
    if waiting:
        return "Waiting for " + _and(label(model, p["role"]) for p in waiting) + ". Answers show up here as they come in."
    add = [a for a in asks if a["kind"] == "add"]
    if add:
        return f"{add[0]['text']}: nobody on the list covers {len(add)} missing topic(s)."
    fu = [a for a in asks if a["kind"] == "follow_up"]
    if fu:
        return (f"All interviews done. {len(fu)} topic(s) still missing: ask a follow-up question to "
                + _and(a["label"] for a in fu) + " (see Results). You can already send to the classifier.")
    return "All interviews done and every topic covered. Review the results, then send to the classifier."


# ---------- exports ----------
def render_csv(res):
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    wr.writeheader()
    wr.writerows(s["row"] for s in res["systems"])
    return buf.getvalue()

def render_index(res, model, target):
    iv = res["interviews"]
    L = [f"# INDEX: {target['name']}", "",
         f"Interviews: {iv['done']} of {iv['total']} done ({', '.join(r['label'] + ' ' + r['status'] for r in iv['roles']) or 'nobody yet'}). "
         "Roles only, no names. Test data.", "",
         f"Systems: {len(res['systems'])} · Conflicts: {len(res['conflicts'])} · Essential topics covered: "
         f"{sum(c['covered'] for c in res['coverage'])} of {sum(c['total'] for c in res['coverage'])}", ""]
    for c in res["coverage"]:
        d = res["details"][c["module"]]
        L += [f"## {c['module']} · {c['name']}", "",
              f"- covered: {c['covered']}/{c['total']}",
              *[f"- missing: {s['id']} → {s['ask']['text']}" for s in c["sections"] if s["essential"] and not s["covered"]],
              f"- conflicts: {d['conflicts']}" + (f" ({', '.join(t['id'] for t in res['conflicts'] if t['module'] == c['module'])})" if d["conflicts"] else ""),
              "- sources: " + (", ".join(f"{s['label']} ({s['weight']})" for s in d["sources"]) or "none"),
              f"- confidence: {d['confidence']:.2f}", f"- feeds: {c['feeds']}"]
        if c["module"] == "SYS":
            ok = d["confidence"] >= 0.5
            L.append(f"- decision rule: SYS {d['confidence']:.2f} {'>=' if ok else '<'} 0.50 → " +
                     ("the classifier may produce a docking plan" if ok else "the classifier refuses the docking plan"))
        L.append("")
    L.append(f"**Next:** {res['next']}")
    return "\n".join(L) + "\n"

def render_tensions(res, target):
    L = [f"# Conflicts: {target['name']}", "", f"{len(res['conflicts'])} conflicts. Both statements are kept; "
         "the working value comes from the stronger source until it is verified. Test data.", ""]
    for t in res["conflicts"]:
        L += [f"## {t['id']} · {t['topic']} ({t['module']} · {t['section']})", "", t["sentence"], ""]
        for side in ("winner", "other"):
            c = t[side]
            L.append(f"- {'working value' if side == 'winner' else 'other view'}: {label_of(t, side)} ({c['bearer_id']}, "
                     f"weight {c['weight']}, {c['kind']}): {t['winner_says' if side == 'winner' else 'other_says']}")
        if t["agree"]:
            L.append(f"- agree with the working value: {', '.join(t['agree'])}")
        if t["decides"]:
            L.append(f"- decides: {t['decides']}")
        L += [f"- how to verify: {t['verify']}", ""]
    return "\n".join(L)

def label_of(t, side):
    return t["winner_role"] if side == "winner" else t["other_role"]

def export(target, model, res=None):
    res = res or results(target, model)
    return {"target_inventory.csv": render_csv(res), "INDEX.md": render_index(res, model, target),
            "tensions.md": render_tensions(res, target)}

def import_payload(target, model, source_url, exported_at, res=None):
    """Body for POST {classifier}/api/inventory/import. Roles only, no names."""
    res = res or results(target, model)
    keys = ("id", "module", "section", "topic", "system", "winner_role", "winner_says", "other_role", "other_says", "verify")
    return {"source": "dd-context-intake", "source_url": source_url,
            "target": {"slug": target["slug"], "name": target["name"]}, "exported_at": exported_at,
            "csv": render_csv(res), "interviews": res["interviews"],
            "conflicts": [{k: t[k] for k in keys} for t in res["conflicts"]],
            "coverage": [{"module": c["module"], "name": c["name"], "covered": c["covered"], "total": c["total"],
                          "missing": [f"{s['label']}: {s['ask']['text']}" for s in c["sections"] if s["essential"] and not s["covered"]]}
                         for c in res["coverage"]]}


# ---------- examples ----------
def load_example(d):
    """dd/examples/<slug>/: people.json = {target: {slug, name}, people: [...]}, interviews/<person-id>.json."""
    d = Path(d)
    meta = json.loads((d / "people.json").read_text())
    t = {**meta["target"], "people": [{"token": None, "status": "done", **p} for p in meta["people"]], "answers": []}
    for p in t["people"]:
        f = d / "interviews" / f"{p['id']}.json"
        if f.exists():
            iv = json.loads(f.read_text())
            t["answers"] += [{"person_id": iv["person_id"], "not_applicable": False, "at": "", **a} for a in iv["answers"]]
    return t


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        sys.exit(__doc__)
    opt = dict(zip(args[1::2], args[2::2]))
    ex = Path(args[0])
    cfg = ROOT / "config.json"
    model = load_model(json.loads(cfg.read_text()).get("model_dir") if cfg.exists() else None)
    target = load_example(ex)
    out = Path(opt.get("--out", ex / "expected"))
    out.mkdir(parents=True, exist_ok=True)
    res = results(target, model)
    for name, text in export(target, model, res).items():
        (out / name).write_text(text)
    print(f"{target['name']}: {res['interviews']['done']}/{res['interviews']['total']} interviews, "
          f"{len(res['systems'])} systems, {len(res['conflicts'])} conflicts → {out}")
    for t in res["conflicts"]:
        print(f"  {t['id']} {t['module']}.{t['section']} {t['system'] or ''}: {t['winner_role']} over {t['other_role']}")
    for a in res["asks"]:
        print(f"  missing {a['module']}.{a['section']}: {a['text']}")
