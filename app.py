#!/usr/bin/env python3
"""DD Context Intake demo server (stdlib only): setup, interviews, live merge, export, send to DD Classifier.
  python3 app.py            serve on http://localhost:8100 (PORT=… to change; DD_MODEL_DIR=… for the model)"""
import csv, datetime, io, json, os, re, secrets, urllib.error, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import engine

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
MODEL = engine.load_model(CFG["model_dir"])  # DD_MODEL_DIR overrides
EXAMPLES = Path(MODEL["dir"]) / "examples" / "campus-crew"
STATE = ROOT / "data" / "state.json"
PREFIX = {"ceo": "ceo", "it_ops_lead": "ops", "field_lead": "field", "finance_lead": "finance", "ucm_deal_lead": "deal"}
DEMO_BEARERS = [  # obviously fictional; names never leave the setup list
    ("ceo-1", "Erika", "Musterfrau", "ceo", "Founder, knows clients and pricing", "interview-02"),
    ("ops-1", "Max", "Mustermann", "it_ops_lead", "Knows payroll export and Zapier end to end", "interview-01"),
    ("field-1", "Paula", "Platzhalter", "field_lead", "Field lead Berlin", "interview-03"),
    ("field-2", "Tim", "Testmann", "field_lead", "Field lead Hamburg and Munich", "interview-04"),
    ("finance-1", "Fiona", "Fiktiv", "finance_lead", "Finance lead, sevDesk and margin report", "interview-05")]


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

def slug(name):
    s = re.sub(r"\(.*?\)|\b(gmbh|ag|ug|se|kg|ltd|inc|llc)\b", "", (name or "").lower())
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "target"

def fresh():
    return {"target": "", "claims": [], "uploads": [], "demo": False,
            "bearers": [{"id": f"{PREFIX[r]}-1", "first": "", "last": "", "email": "", "role": r, "note": "",
                         "token": None, "status": "not invited"} for r in engine.TARGET_ROLES]}

def load():
    return json.loads(STATE.read_text()) if STATE.exists() else fresh()

def save(st):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1, ensure_ascii=False))
    return st

def new_id(role, bearers):
    taken = {b["id"] for b in bearers}
    return next(f"{PREFIX.get(role, role)}-{n}" for n in range(1, 99) if f"{PREFIX.get(role, role)}-{n}" not in taken)

def find(st, key, val):
    b = next((b for b in st["bearers"] if b.get(key) == val), None)
    if not b:
        raise KeyError("unknown bearer or link")
    return b


# ---------- actions ----------
def save_bearer(st, body):
    role = body.get("role", "it_ops_lead")
    if role not in MODEL["weights"]["roles"]:
        raise ValueError(f"unknown role {role}")
    if body.get("id"):
        b = find(st, "id", body["id"])
        if role != b["role"]:
            if any(c["bearer_id"] == b["id"] for c in st["claims"]):
                raise ValueError("this person already answered; remove and re-add to change the role")
            b["role"], b["id"] = role, new_id(role, st["bearers"])
    else:
        b = {"id": new_id(role, st["bearers"]), "role": role, "token": None, "status": "not invited"}
        st["bearers"].append(b)
    b.update({k: str(body.get(k, b.get(k, ""))).strip() for k in ("first", "last", "email", "note")})
    return save(st)

def invite(st):
    for b in st["bearers"]:
        if not b.get("token"):
            b["token"], b["status"] = secrets.token_hex(3), "invited"
    return save(st)

def prefill():
    st = {"target": "Campus Crew GmbH (test)", "uploads": [], "demo": True,
          "claims": engine.load_claims(EXAMPLES), "bearers": []}
    for bid, first, last, role, note, src in DEMO_BEARERS:
        st["bearers"].append({"id": bid, "first": first, "last": last, "email": f"{first}.{last}@example.com".lower(),
                              "role": role, "note": note, "token": secrets.token_hex(3), "status": "done",
                              "source_id": src})
    return save(st)

def upload(st, filename, data):
    if not filename.lower().endswith((".csv", ".txt")):
        raise ValueError("CSV only in this demo (export the tool or licence list as .csv)")
    text = data.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text), delimiter=";" if text.count(";") > text.count(",") else ","))
    head = [h.strip().lower().replace(" ", "_") for h in rows[0]] if rows else []
    if "name" not in head:
        raise ValueError("need a header row with at least a 'name' column (classifier column names)")
    src = f"upload-{len(st['uploads']) + 1:02d}"
    n = 0
    for raw in rows[1:]:
        f = {k: v.strip() for k, v in zip(head, raw) if k in engine.CSV_FIELDS[:9]}
        if not f.get("name"):
            continue
        f.setdefault("purpose", f.get("notes") or f.get("category", ""))
        st["claims"].append({"target": slug(st["target"]), "module": "SYS", "section": "system_list", "value": f["name"],
                             "value_type": "text", "kind": "data", "source_role": "upload", "bearer_id": "upload",
                             "source_id": src, "quote": f"row from {Path(filename).name}", "systems_mentioned": [f["name"]],
                             "timestamp": now(), "fields": f})
        n += 1
    st["uploads"].append({"source_id": src, "file": Path(filename).name, "rows": n, "at": now()})
    save(st)
    return {"ok": True, "rows": n, "source_id": src}

def answer(st, token, body):
    b = find(st, "token", token)
    q = next((q for q in engine.questions(MODEL, b["role"]) if (q["module"], q["section"]) ==
              (body.get("module"), body.get("section"))), None)
    if not q:
        raise ValueError("unknown question")
    na, text = bool(body.get("not_applicable")), str(body.get("answer", "")).strip()
    if not na and not text:
        raise ValueError("empty answer")
    if not b.get("source_id"):
        used = [int(c["source_id"][10:]) for c in st["claims"] if re.fullmatch(r"interview-\d+", c["source_id"])]
        b["source_id"] = f"interview-{max(used + [0]) + 1:02d}"
    st["claims"].append({"target": slug(st["target"]), "module": q["module"], "section": q["section"],
                         "value": "not applicable" if na else text, "value_type": "text", "kind": "estimate",
                         "source_role": b["role"], "bearer_id": b["id"], "source_id": b["source_id"],
                         "quote": "doesn't apply" if na else text[:200], "systems_mentioned": [],
                         "timestamp": now(), "not_applicable": na, "extraction": "stub: raw answer, not normalised"})
    b["status"] = "done" if b["status"] == "done" else "started"
    save(st)
    return interview(st, token)

def interview(st, token):
    b = find(st, "token", token)
    if b["status"] == "invited":
        b["status"] = "started"
        save(st)
    own = [c for c in st["claims"] if c["bearer_id"] == b["id"]]
    return {"target": st["target"], "slug": slug(st["target"]), "status": b["status"],
            "bearer": {"id": b["id"], "role": b["role"], "label": engine.role_label(MODEL, b["role"])},
            "weights": MODEL["weights"]["roles"][b["role"]], "questions": engine.questions(MODEL, b["role"]),
            "answered": [{"module": c["module"], "section": c["section"], "value": c["value"]} for c in own],
            "coverage": engine.coverage(own, MODEL), "modules": [{"id": m["id"], "name": m["name"]} for m in MODEL["modules"]]}

def merged(st):
    m = engine.merge(st["claims"], MODEL)
    for t in m["tensions"]:
        t.pop("_claims", None)
    return m

def export(st):
    files = engine.export(st["claims"], MODEL, slug(st["target"]))
    out = ROOT / "out" / slug(st["target"])
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text)
    return {"dir": str(out.relative_to(ROOT)), "files": files}

def send_to_classifier(st):
    base, user = CFG["classifier_url"].rstrip("/"), CFG["classifier_user"]
    body = engine.export(st["claims"], MODEL, slug(st["target"]))["target_inventory.csv"].encode()
    if body.count(b"\n") < 2:
        return {"ok": False, "error": "No systems yet: run an interview with SYS answers, upload a tool list or pre-fill demo data."}
    try:
        req = urllib.request.Request(base + "/api/login", json.dumps({"user_id": user}).encode(),
                                     {"Content-Type": "application/json"})
        tok = json.load(urllib.request.urlopen(req, timeout=5))["token"]
        req = urllib.request.Request(base + "/api/inventory/upload", body, {"X-Session": tok, "Content-Type": "text/csv",
                                     "X-Filename": f"{slug(st['target'])}_target_inventory.csv"})
        meta = json.load(urllib.request.urlopen(req, timeout=15))
        return {"ok": True, "rows": meta.get("rows"), "warnings": meta.get("warnings", []), "as": user, "open": base}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"DD Classifier answered {e.code}: {e.read().decode(errors='replace')[:300]}"}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"ok": False, "error": f"DD Classifier not reachable at {base} ({getattr(e, 'reason', e)}). "
                "Start the classifier: cd ../dd-registry-classifier && python3 app.py"}

def public_state(st):
    return {"target": st["target"], "slug": slug(st["target"]), "demo": st.get("demo"), "uploads": st["uploads"],
            "bearers": st["bearers"], "claims": len(st["claims"]), "classifier_url": CFG["classifier_url"],
            "model_dir": MODEL["dir"], "roles": {r: v["label"] for r, v in MODEL["weights"]["roles"].items()},
            "answered": {b["id"]: sum(c["bearer_id"] == b["id"] for c in st["claims"]) for b in st["bearers"]}}


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")

    def route(self, method):
        path, st = urllib.parse.unquote(self.path.split("?")[0]), load()
        try:
            if method == "GET":
                if path == "/" or path.startswith("/i/"):
                    return self.send(200, (ROOT / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
                if path == "/assets/icon.svg":
                    return self.send(200, (ROOT / "assets" / "icon.svg").read_bytes(), "image/svg+xml")
                if path == "/api/state":
                    return self.send(200, public_state(st))
                if path == "/api/merged":
                    return self.send(200, merged(st))
                m = re.fullmatch(r"/api/interview/(\w+)", path)
                if m:
                    return self.send(200, interview(st, m[1]))
                m = re.fullmatch(r"/api/download/(target_inventory\.csv|INDEX\.md|tensions\.md)", path)
                if m:
                    return self.send(200, export(st)["files"][m[1]].encode(), "text/plain; charset=utf-8",
                                     {"Content-Disposition": f'attachment; filename="{m[1]}"'})
                return self.send(404, {"error": "not found"})
            if path == "/api/upload":
                return self.send(200, upload(st, urllib.parse.unquote(self.headers.get("X-Filename", "")), self.body()))
            b = json.loads(self.body() or b"{}")
            m = re.fullmatch(r"/api/(interview/(\w+)|bearers/([\w-]+)/delete)(/done)?", path)
            if m and m[2] and m[4]:
                find(st, "token", m[2])["status"] = "done"
                return self.send(200, interview(save(st), m[2]))
            if m and m[2]:
                return self.send(200, answer(st, m[2], b))
            if m and m[3]:
                find(st, "id", m[3])
                st["bearers"] = [x for x in st["bearers"] if x["id"] != m[3]]
                st["claims"] = [c for c in st["claims"] if c["bearer_id"] != m[3]]
                return self.send(200, public_state(save(st)))
            if path == "/api/target":
                st["target"] = str(b.get("name", "")).strip()
                return self.send(200, public_state(save(st)))
            acts = {"/api/bearers": lambda: public_state(save_bearer(st, b)), "/api/invite": lambda: public_state(invite(st)),
                    "/api/demo": lambda: public_state(prefill()), "/api/reset": lambda: public_state(save(fresh())),
                    "/api/export": lambda: export(st), "/api/send": lambda: send_to_classifier(st)}
            if path in acts:
                return self.send(200, acts[path]())
            self.send(404, {"error": "not found"})
        except (ValueError, KeyError) as e:
            self.send(400, {"error": str(e).strip("'\"")})
        except Exception as e:  # keep the demo alive and show the problem
            self.send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", CFG["port"]))
    print(f"DD Context Intake on http://localhost:{port}  (model: {MODEL['dir']})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
