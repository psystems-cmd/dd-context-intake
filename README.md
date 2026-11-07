<img src="assets/icon.svg" width="64" height="64" alt="">

# DD Context Intake (demo)

Several people at an acquisition target each run a short role-based interview; the **DD Context Engine** merges their claims into one weighted picture (coverage, confidence, tensions) and exports a `target_inventory.csv` the [DD Classifier](../dd-registry-classifier) runs against UCM's registry. **Test data only.**

```
python3 app.py          # Python 3.9+, stdlib only → http://localhost:8100   (PORT=… / DD_MODEL_DIR=… to change)
python3 -m unittest -v  # engine tests, incl. "the classifier's intake accepts the CSV without edits"
python3 engine.py ../structured-context-model/dd/examples/campus-crew --bearers ops-1,ceo-1 --out /tmp/out   # CLI merge
```

## 1-minute demo

1. **Setup** → *Pre-fill with demo data*: target "Campus Crew GmbH (test)", five fictional bearers (CEO, IT/Ops lead, 2 Field leads, Finance lead), all interviews done. Optionally upload `samples/licence_export.csv` (rows become `data` claims, weight 1.0).
2. **Merged picture**: 8 DD modules with confidence, essential coverage, sources with weights; **3 tensions** (student data, payouts, margin) side by side with suggested winner and how to verify.
3. **Export**: writes `out/campus-crew/` (`target_inventory.csv`, `INDEX.md`, `tensions.md`), previews and downloads them.
4. **Send to DD Classifier**: logs in to the classifier as `target1_it` and uploads the CSV (start it first: `cd ../dd-registry-classifier && python3 app.py`).
5. **Open the classifier** (http://localhost:8000), log in as Platform Lead, run DD → docking plan for the merged inventory.

To show the multi-source story: *Reset*, enter a target, *Send invites*, open the IT/Ops link (`/i/<slug>/<token>`) and answer a few questions, then the CEO link: the picture grows; nothing is overwritten.

## How it links

| Piece | Where | Role |
| --- | --- | --- |
| Model | `../structured-context-model/dd/` (`weights.yaml`, `schemas/dd-modules.json`, `examples/campus-crew/`) | Read at startup (`model_dir` in `config.json`, `DD_MODEL_DIR` overrides). Never copied. |
| Engine | `engine.py` | Pure functions: weights parser, merge per section, tension rules, confidence, next-role hint, the 3 exports, scripted questions. Also produces `examples/campus-crew/expected/`. |
| App | `app.py`, `static/index.html` | Setup (bearers, invites, upload, demo, reset), interview by personal link, merged view, export, send. State in `data/state.json` (gitignored). |
| Classifier | `../dd-registry-classifier` | Downstream consumer: `POST /api/login` + `POST /api/inventory/upload` (raw CSV, `X-Filename`). |

Privacy: names live only in the setup list (`data/state.json`); claims and exports carry `bearer_id` and role only.

## Live vs stubbed

| Step | In this demo |
| --- | --- |
| Setup, invites (links, no emails), upload | live |
| Interview | **scripted**: questions generated from `dd-modules.json` + role weights (deep modules first, essential sections); each answer is stored as one raw text claim (`estimate`, no norm) |
| Extraction | **stubbed**: demo uses the hand-written claims in `examples/campus-crew/claims/` |
| Merge, confidence, tensions | live (`engine.py`) |
| Export + send to DD Classifier | live |

## What's next

- Real extraction: run `dd/prompts/extraction.md` on a cheap EU model via Lyceum to turn transcripts into normalised claims (norms, PROC fields, systems).
- Adaptive interviewer behind the Figma front end (this page stands in for it).
- Classifier passes `notes`, `confidence` and `sources` through to its result table (today its intake ignores the last two columns with a warning).
