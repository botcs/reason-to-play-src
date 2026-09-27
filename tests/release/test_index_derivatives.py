import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/release/index_derivatives.py"
EXAMPLE_RULES = SCRIPT.parents[2] / "docs/release/asset-rules.example.json"
spec = importlib.util.spec_from_file_location("index_derivatives", SCRIPT)
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.rules = index.load_rules(EXAMPLE_RULES)

    def row(self, key):
        return {"bucket": "test", "key": key, "size_bytes": 17, "etag": "opaque-9"}

    def test_pipe_paths_are_reversible_and_have_structured_metadata(self):
        suffix = (
            "copied-reasoning|model|elaborate|sub-01|bait_vgfmri3/a% b.replay.json.gz"
        )
        row = index.classify(self.row("example-replays/" + suffix), self.rules)
        self.assertEqual(unquote(row["release_path"]), "replays/human/" + suffix)
        self.assertNotIn("|", row["release_path"])
        self.assertEqual(row["subject"], "sub-01")
        self.assertEqual(row["rationale_mode"], "copied-reasoning")
        self.assertEqual(row["game_path_id"], "bait_vgfmri3")
        self.assertEqual(row["etag"], "opaque-9")
        self.assertEqual(row["provenance_status"], "object-metadata-only")

    def test_unknown_and_excluded_material_cannot_be_selected(self):
        self.assertEqual(
            index.classify(self.row("new-experiment/a.pt"), self.rules)["disposition"],
            "review",
        )
        for key in (
            "example-excluded/a.pt",
            "fmriprep/.DS_Store",
            "fmriprep/.git/config",
            "fmriprep/.env",
            "llm_replay/../escape",
            "fmriprep/a\\b",
            "fmriprep/folder/",
        ):
            with self.subTest(key=key):
                self.assertEqual(
                    index.classify(self.row(key), self.rules)["disposition"], "exclude"
                )

    def test_manifest_rejects_collisions_before_publishing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rules = {
                "schema_version": 1,
                "prefixes": {
                    p: {
                        "release_root": "same",
                        "component": "test",
                        "disposition": "candidate",
                        "reason": "fixture",
                    }
                    for p in ("first", "second")
                },
            }
            (root / "rules.json").write_text(json.dumps(rules))
            (root / "input.jsonl").write_text(
                "\n".join(
                    json.dumps(self.row(p + "/a.pt")) for p in ("first", "second")
                )
            )
            args = argparse.Namespace(
                input=str(root / "input.jsonl"),
                output=str(root / "output"),
                rules=str(root / "rules.json"),
            )
            with self.assertRaisesRegex(ValueError, "collision"):
                index.build(args)
            self.assertFalse((root / "output/objects.jsonl.gz").exists())

    def test_summary_and_candidates_account_for_all_source_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keys = ["example-replays/a.gz", "example-excluded/b.pt", "unknown/c.pt"]
            (root / "input.jsonl").write_text(
                "\n".join(json.dumps(self.row(k)) for k in keys)
            )
            args = argparse.Namespace(
                input=str(root / "input.jsonl"),
                output=str(root / "output"),
                rules=str(EXAMPLE_RULES),
            )
            index.build(args)
            summary = json.loads((root / "output/summary.json").read_text())
            self.assertEqual(summary["objects"], 3)
            self.assertEqual(summary["bytes"], 51)
            self.assertEqual(
                sum(d["objects"] for d in summary["dispositions"].values()), 3
            )
            with gzip.open(root / "output/release-candidates.jsonl.gz", "rt") as handle:
                selected = [json.loads(line) for line in handle]
            self.assertEqual([r["key"] for r in selected], [keys[0]])

    def test_cli_requires_rules_and_bucket_before_any_io(self):
        for args, option in (
            (["build", "--input", "missing.jsonl", "--output", "unused"], "--rules"),
            (["scan", "--output", "unused"], "--bucket"),
        ):
            with (
                self.subTest(option=option),
                patch.object(sys, "argv", [str(SCRIPT), *args]),
            ):
                with self.assertRaises(SystemExit) as error:
                    index.main()
                self.assertEqual(error.exception.code, 2)

    def test_scan_consumes_every_page_including_empty_page(self):
        class Date:
            def isoformat(self):
                return "2026-09-27T00:00:00+00:00"

        class Client:
            def get_paginator(self, operation):
                self.operation = operation
                return self

            def paginate(self, **kwargs):
                return iter(
                    [
                        {},
                        {
                            "Contents": [
                                {
                                    "Key": "a",
                                    "Size": 1,
                                    "LastModified": Date(),
                                    "ETag": '"multipart-8"',
                                }
                            ]
                        },
                        {
                            "Contents": [
                                {
                                    "Key": "b",
                                    "Size": 2,
                                    "LastModified": Date(),
                                    "ETag": '"opaque"',
                                }
                            ]
                        },
                    ]
                )

        client = Client()
        result = list(index.scan_objects(client, "test"))
        self.assertEqual(client.operation, "list_objects_v2")
        self.assertEqual([r["key"] for r in result], ["a", "b"])
        self.assertEqual(result[0]["etag"], "multipart-8")


if __name__ == "__main__":
    unittest.main()
