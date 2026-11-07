"""python3 -m unittest -v   (reads the model + examples from ../structured-context-model/dd)"""
import importlib, json, re, sys, unittest
from pathlib import Path
import engine

ROOT = Path(__file__).resolve().parent
MODEL = engine.load_model(json.loads((ROOT / "config.json").read_text())["model_dir"])
EX = Path(MODEL["dir"]) / "examples" / "campus-crew"
TWO = engine.load_claims(EX, ["ops-1", "ceo-1"])
ALL = engine.load_claims(EX)
CLASSIFIER = ROOT.parent / "dd-registry-classifier"


class Merge(unittest.TestCase):
    def test_two_interviews_three_tensions_ops_wins(self):
        m = engine.merge(TWO, MODEL)
        got = {(t["module"], t["section"]): t for t in m["tensions"]}
        self.assertEqual(set(got), {("DATA", "data_quality"), ("PROC", "timesheet_to_payout"),
                                    ("COST", "margin_by_placement_type")})
        for t in got.values():
            self.assertEqual(t["winner"]["bearer_id"], "ops-1")
            self.assertEqual(t["other"]["bearer_id"], "ceo-1")

    def test_full_demo_three_tensions_five_contributors(self):
        m = engine.merge(ALL, MODEL)
        self.assertEqual(len(m["tensions"]), 3)
        self.assertEqual(m["contributors"], ["ceo-1", "field-1", "field-2", "finance-1", "ops-1"])
        self.assertTrue(all(t["other"]["bearer_id"] == "ceo-1" for t in m["tensions"]))

    def test_confidence_and_next_role(self):
        m = engine.merge(TWO, MODEL)
        c = {k: v["confidence"] for k, v in m["modules"].items()}
        self.assertGreaterEqual(c["SYS"], 0.6)
        self.assertLess(c["PROC"], c["SYS"])
        self.assertLess(c["COST"], c["SYS"])
        self.assertEqual(m["next_role"]["role"], "finance_lead")

    def test_disagreement_rules(self):
        d = lambda a, b: engine.disagree({"norm": a}, {"norm": b}, 20)
        self.assertTrue(d({"number": 35}, {"range": [22, 38]}))
        self.assertFalse(d({"number": 35}, {"range": [32, 38]}))
        self.assertTrue(d({"number": 100}, {"number": 130}))
        self.assertFalse(d({"number": 100}, {"number": 110}))
        self.assertTrue(d({"state": "manual"}, {"state": "automated"}))
        self.assertTrue(d({"items": ["Excel"]}, {"denies": ["excel"]}))
        self.assertFalse(engine.disagree({"norm": {"state": "a"}}, {}, 20))

    def test_not_applicable_needs_deep_role(self):
        sec = {"id": "x", "type": "qualitative", "essential": True}
        na = lambda role: {"not_applicable": True, "source_role": role, "value": "not applicable"}
        self.assertTrue(engine.has_content(na("it_ops_lead"), sec, MODEL, "SYS"))
        self.assertFalse(engine.has_content(na("ceo"), sec, MODEL, "SYS"))


class Export(unittest.TestCase):
    def test_csv_header_exact(self):
        csv_text = engine.export(TWO, MODEL, "campus-crew")["target_inventory.csv"]
        self.assertEqual(csv_text.splitlines()[0],
                         "name,vendor,category,users,cost_per_year,last_used,owner_role,holds_data,notes,confidence,sources")
        self.assertIn("(tension)", csv_text)
        self.assertGreaterEqual(len(csv_text.splitlines()) - 1, 11)

    def test_expected_files_match_engine(self):
        for name, text in engine.export(TWO, MODEL, "campus-crew").items():
            self.assertEqual((EX / "expected" / name).read_text(), text, name)

    def test_no_names_in_claims_or_exports(self):
        sys.path.insert(0, str(ROOT))
        app = importlib.import_module("app")
        names = [n for b in app.DEMO_BEARERS for n in (b[1], b[2])]
        blob = json.dumps(ALL) + "".join(engine.export(ALL, MODEL, "campus-crew").values())
        for n in names:
            self.assertIsNone(re.search(r"\b%s\b" % n, blob), n)

    @unittest.skipUnless((CLASSIFIER / "intake.py").exists(), "dd-registry-classifier not checked out next to this repo")
    def test_classifier_accepts_without_edits(self):
        sys.path.insert(0, str(CLASSIFIER))
        intake = importlib.import_module("intake")
        for claims in (TWO, ALL):
            data = engine.export(claims, MODEL, "campus-crew")["target_inventory.csv"].encode()
            rows, meta = intake.map_table(intake.table_from_bytes("x.csv", data))
            self.assertEqual(len(rows), len(data.decode().strip().splitlines()) - 1)
            self.assertFalse([w for w in meta["warnings"] if "skipped" in w])
            self.assertEqual(meta["missing"], [])


if __name__ == "__main__":
    unittest.main()
