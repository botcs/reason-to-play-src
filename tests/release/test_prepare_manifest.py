"""Release integrity contracts: snapshots are not checksums or staged payloads."""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from urllib.parse import unquote

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/release/prepare_manifest.py"
spec = importlib.util.spec_from_file_location("prepare_manifest", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_cli_requires_explicit_policy_before_reading_inventory(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "build", "--inventory", "missing.jsonl", "--output", "unused"],
    )
    with pytest.raises(SystemExit) as error:
        release.main()
    assert error.value.code == 2
    assert "--policy" in capsys.readouterr().err


def test_published_example_builds_offline_and_preserves_selection_states(tmp_path):
    example = SCRIPT.parents[2] / "docs/release/policy.example.json"
    source = tmp_path / "input.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "bucket": "example-bucket",
                    "key": key,
                    "size_bytes": 1,
                    "etag": "example-etag",
                }
            )
            for key in (
                "example-replays/run|one.json.gz",
                "example-features/block.npz",
                "example-archive/old.npz",
                "example-excluded/workspace.txt",
                "unknown/item.json",
            )
        )
    )
    output = tmp_path / "prepared"
    release.build(argparse.Namespace(inventory=source, policy=example, output=output))
    records = list(release.read_rows(output / "manifest.jsonl.gz"))
    assert [r["decision"] for r in records] == ["include", "optional"]
    assert records[0]["release_path"] == "replays/human/run%7Cone.json.gz"
    summary = json.loads((output / "summary.json").read_text())
    assert sum(g["objects"] for g in summary["groups"]) == 5
    assert {g["decision"] for g in summary["groups"]} == {
        "include",
        "optional",
        "defer",
        "exclude",
    }


@pytest.fixture
def policy():
    return {
        "schema_version": 1,
        "release_id": "test",
        "target_repository": "author/test",
        "derivative_license": "pending",
        "rules": [
            {
                "id": "replays",
                "source_prefix": "llm_replay/",
                "release_root": "replays/human",
                "decision": "include",
                "tier": "core",
                "component": "human-replay",
                "reason": "fixture",
            }
        ],
    }


@pytest.fixture
def row(policy):
    source = {
        "bucket": "test",
        "key": "llm_replay/sub-13/a|b%.gz",
        "etag": "opaque-2",
        "size_bytes": 5,
        "last_modified": "2026-09-27T00:00:00+00:00",
    }
    return release.make_row(source, policy["rules"][0], policy)


class Client:
    def __init__(self, data=b"hello", **metadata):
        self.data = data
        self.metadata = dict(
            ContentLength=5,
            ETag='"opaque-2"',
            LastModified=datetime(2026, 9, 27, tzinfo=timezone.utc),
            VersionId="version-one",
        )
        self.metadata.update(metadata)
        self.calls = []

    def head_object(self, **kwargs):
        self.calls.append(("head", kwargs))
        return self.metadata

    def get_object(self, **kwargs):
        self.calls.append(("get", kwargs))
        self.body = io.BytesIO(self.data)
        return {**self.metadata, "Body": self.body}


def test_pin_checks_inventory_generation_and_preserves_identity(row):
    client = Client()
    pinned = release.pin_source(row, client)
    assert pinned["artifact_id"] == row["artifact_id"]
    assert pinned["source"]["version_id"] == "version-one"
    assert pinned["payload"]["sha256"] is None
    assert client.calls[0][1]["IfMatch"] == "opaque-2"
    assert row["source"]["verification"] == "listed-only"
    for drift in (
        {"ContentLength": 6},
        {"ETag": '"replacement"'},
        {"LastModified": datetime(2026, 9, 28, tzinfo=timezone.utc)},
    ):
        with pytest.raises(ValueError, match="changed since inventory"):
            release.pin_source(row, Client(**drift))


def test_stage_downloads_pinned_version_and_hashes_actual_bytes(tmp_path, row):
    client = Client()
    pinned = release.pin_source(row, client)
    sibling = release.destination(tmp_path, row["release_path"] + ".partial")
    sibling.write_text("another legitimate artifact")
    result = release.stage_object(pinned, client, tmp_path)
    assert client.calls[-1][1]["VersionId"] == "version-one"
    assert "IfMatch" not in client.calls[-1][1]
    assert result["payload"] == {
        "sha256": hashlib.sha256(b"hello").hexdigest(),
        "validation": "bytes-verified",
    }
    assert (tmp_path / row["release_path"]).read_bytes() == b"hello"
    assert sibling.read_text() == "another legitimate artifact"
    assert client.body.closed
    calls = len(client.calls)
    release.stage_object(result, client, tmp_path)
    assert (
        len(client.calls) == calls
    )  # Cache is checked by bytes + SHA, never existence alone.


def test_unversioned_source_keeps_conditional_get(tmp_path, row):
    client = Client(VersionId="null")
    pinned = release.pin_source(row, client)
    assert pinned["source"]["verification"] == "head-verified-unversioned"
    assert pinned["source"]["version_id"] is None
    release.stage_object(pinned, client, tmp_path)
    assert client.calls[-1][1]["IfMatch"] == "opaque-2"


@pytest.mark.parametrize(
    "data,metadata,error",
    [
        (b"hel", {}, "Truncated"),
        (b"hello", {"ETag": "changed"}, "metadata differs"),
        (b"hello", {"ContentLength": 6}, "metadata differs"),
    ],
)
def test_bad_download_never_replaces_existing_payload(
    tmp_path, row, data, metadata, error
):
    pinned = release.pin_source(row, Client())
    target = release.destination(tmp_path, row["release_path"])
    target.write_bytes(b"previous")
    client = Client(data=data, **metadata)
    with pytest.raises(ValueError, match=error):
        release.stage_object(pinned, client, tmp_path)
    assert target.read_bytes() == b"previous"
    assert client.body.closed
    assert not list(target.parent.glob(".stage-*"))


def test_checksum_failure_and_unverified_source_are_not_publishable(tmp_path, row):
    with pytest.raises(ValueError, match="Freeze/verify"):
        release.stage_object(row, Client(), tmp_path)
    pinned = release.pin_source(row, Client())
    pinned["payload"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        release.stage_object(pinned, Client(), tmp_path)
    assert not (tmp_path / row["release_path"]).exists()


def test_paths_are_reversible_and_staging_rejects_symlink_escape(tmp_path, row):
    assert unquote(row["release_path"]) == "replays/human/sub-13/a|b%.gz"
    assert row["metadata"]["subject"] == "sub-13"
    outside = tmp_path / "outside"
    outside.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "replays").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        release.stage_object(release.pin_source(row, Client()), Client(), staging)
    assert not list(outside.iterdir())


def test_build_is_deterministic_accounts_for_exclusions_and_blocks_collisions(
    tmp_path, row, policy
):
    objects = [
        row["source"],
        {**row["source"], "key": "llm_replay/.aws/credentials"},
        {**row["source"], "key": "unknown/secret-not-public"},
    ]
    source = tmp_path / "input.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in objects))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        release.build(
            argparse.Namespace(inventory=source, policy=policy_path, output=output)
        )
    for filename in ("manifest.jsonl.gz", "summary.json", "coverage.csv"):
        assert (outputs[0] / filename).read_bytes() == (
            outputs[1] / filename
        ).read_bytes()
    summary = json.loads((outputs[0] / "summary.json").read_text())
    assert summary["selected_objects"] == 1
    assert sum(g["objects"] for g in summary["groups"]) == 3
    assert "secret-not-public" not in (outputs[0] / "summary.json").read_text()
    rows = list(release.read_rows(outputs[0] / "manifest.jsonl.gz"))
    assert rows[0]["source"]["verification"] == "listed-only"
    assert rows[0]["license"] == "pending"
    policy["rules"].append(
        {**policy["rules"][0], "id": "duplicate", "source_prefix": "another/"}
    )
    objects.append({**row["source"], "key": "another/sub-13/a|b%.gz"})
    source.write_text("\n".join(json.dumps(r) for r in objects))
    policy_path.write_text(json.dumps(policy))
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="collision"):
        release.build(
            argparse.Namespace(inventory=source, policy=policy_path, output=failed)
        )
    assert not (failed / "manifest.jsonl.gz").exists()


def test_validation_does_not_confuse_pinning_with_payload_integrity(tmp_path, row):
    path = tmp_path / "manifest.jsonl"
    pinned = release.pin_source(row, Client())
    path.write_text(json.dumps(pinned))
    args = argparse.Namespace(
        manifest=path, tier=None, component=None, require_staged=True
    )
    with pytest.raises(ValueError, match="Payload not verified"):
        release.validate(args)
    args.require_staged = False
    args.component = ["missing"]
    with pytest.raises(ValueError, match="Selection is empty"):
        release.validate(args)
    broken = copy.deepcopy(pinned)
    broken["source"]["version_id"] = None
    with pytest.raises(ValueError, match="missing its version"):
        release.check_row(broken)


def test_manifest_writer_preserves_legitimate_partial_filename(tmp_path):
    target = tmp_path / "output.jsonl.gz"
    neighbor = tmp_path / "output.jsonl.gz.partial"
    neighbor.write_bytes(b"legitimate payload")
    with release.write_rows(target) as handle:
        release.emit(handle, {"valid": True})
    assert list(release.read_rows(target)) == [{"valid": True}]
    assert neighbor.read_bytes() == b"legitimate payload"


@pytest.mark.parametrize(
    "metadata",
    [
        {"LastModified": datetime(2026, 9, 28, tzinfo=timezone.utc)},
        {"VersionId": "another-version"},
    ],
)
def test_get_must_match_frozen_generation(tmp_path, row, metadata):
    pinned = release.pin_source(row, Client())
    with pytest.raises(ValueError, match="metadata differs"):
        release.stage_object(pinned, Client(**metadata), tmp_path)
    assert not (tmp_path / row["release_path"]).exists()


@pytest.mark.parametrize("filename", ["source.jsonl", "output.jsonl"])
def test_staging_cannot_overwrite_its_manifest(tmp_path, row, filename):
    pinned = release.pin_source(row, Client())
    pinned["release_path"] = filename
    source = tmp_path / "source.jsonl"
    original = json.dumps(pinned)
    source.write_text(original)
    args = argparse.Namespace(
        manifest=source,
        output=tmp_path / "output.jsonl",
        directory=tmp_path,
        tier=None,
        component=None,
        max_bytes=10,
        region="eu-west-2",
    )
    with pytest.raises(ValueError, match="paths collide"):
        release.stage(args)
    assert source.read_text() == original


def test_rechecking_frozen_manifest_uses_pinned_generation(row):
    client = Client()
    pinned = release.pin_source(row, client)
    release.pin_source(pinned, client)
    assert client.calls[-1][1]["VersionId"] == "version-one"
    with pytest.raises(ValueError, match="changed since inventory"):
        release.pin_source(pinned, Client(VersionId="replacement"))


def test_verified_hash_merge_keeps_current_license_and_rejects_wrong_generation(
    tmp_path, row
):
    pinned = release.pin_source(row, Client())
    staged = release.stage_object(pinned, Client(), tmp_path / "stage")
    staged_file = tmp_path / "staged.jsonl"
    staged_file.write_text(json.dumps(staged))
    pinned["license"] = "MIT"
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(pinned))
    output = tmp_path / "merged.jsonl"
    args = argparse.Namespace(manifest=manifest, verified=[staged_file], output=output)
    release.merge_verified(args)
    result = next(release.read_rows(output))
    assert result["license"] == "MIT"
    assert result["payload"]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    staged["source"]["version_id"] = "wrong-version"
    staged_file.write_text(json.dumps(staged))
    previous = output.read_bytes()
    with pytest.raises(ValueError, match="source generation"):
        release.merge_verified(args)
    assert output.read_bytes() == previous


def test_resume_freeze_preserves_original_version_check_time(
    tmp_path, row, monkeypatch
):
    import boto3

    pinned = release.pin_source(row, Client())
    cache = tmp_path / "old.jsonl"
    cache.write_text(json.dumps(pinned))
    source = tmp_path / "new.jsonl"
    row["license"] = "MIT"
    source.write_text(json.dumps(row))

    class NoNetwork:
        def head_object(self, **kwargs):
            pytest.fail("Matching immutable version must be reused")

    monkeypatch.setattr(boto3, "client", lambda *a, **kw: NoNetwork())
    output = tmp_path / "verified.jsonl"
    args = argparse.Namespace(
        manifest=source,
        reuse_pins=cache,
        output=output,
        tier=None,
        component=None,
        workers=2,
        region="eu-west-2",
    )
    release.freeze(args)
    actual = next(release.read_rows(output))
    assert actual["source"]["verified_at"] == pinned["source"]["verified_at"]
    assert actual["license"] == "MIT"


def test_mixed_dataset_assembly_preserves_identity_and_uses_generated_sizes(
    tmp_path, row
):
    import pyarrow.parquet as pq

    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row))
    mapping = tmp_path / "map.jsonl"
    mapping.write_text(
        json.dumps(
            {
                "artifact_id": row["artifact_id"],
                "action": "include",
                "release_path": "features/lrm/model/elaborate/all/sub-13/game.pt",
                "selection_reason": "Selected model input",
                "metadata": {
                    "model_family": "lrm",
                    "model_id": "model",
                    "condition": "elaborate",
                    "timebase": "selected-prompt-step",
                    "context_fraction": 0.5,
                },
            }
        )
    )
    payload = tmp_path / "generated.json"
    payload.write_text('{"complete": true}\n')
    local = tmp_path / "local.jsonl"
    local.write_text(
        json.dumps(
            {
                "local_path": str(payload),
                "release_path": "behavior/ddqn/episode-history.json",
                "component": "baseline_behavior",
                "license": "MIT",
                "selection_reason": "Recorded episodes",
                "metadata": {"model_family": "ddqn", "model_id": "ddqn"},
                "provenance": {"input_size_bytes": 999},
            }
        )
    )
    args = argparse.Namespace(
        source_manifest=source,
        path_map=[mapping],
        local_files=[local],
        human_manifest=None,
        human_directory=None,
        enrichments=None,
        release_id="test-v2",
        output=tmp_path / "dataset",
        audit_output=tmp_path / "private",
        workers=2,
        max_bytes=1000,
    )
    release.assemble(args)
    manifest = args.output / "manifest.jsonl.gz"
    rows = list(release.read_rows(manifest))
    generated, original = rows
    assert original["artifact_id"] == row["artifact_id"]
    assert original["artifact_id_scheme"] == "frozen-source-identity"
    assert original["source"]["key"] == row["source"]["key"]
    assert original["payload"]["sha256"] is None
    assert (
        generated["artifact_id"]
        == "sha256:" + hashlib.sha256(payload.read_bytes()).hexdigest()
    )
    assert generated["payload"]["size_bytes"] == payload.stat().st_size
    assert generated["source_uri"] is None
    assert (
        args.output / generated["release_path"]
    ).read_bytes() == payload.read_bytes()
    table = pq.read_table(
        args.output / "catalog/files/planned_files.parquet"
    ).to_pylist()
    assert table[0]["size_bytes"] == payload.stat().st_size
    assert table[1]["context_fraction"] == 0.5
    assert all(r["published"] is False for r in table)
    assert str(tmp_path) not in gzip_text(manifest)
    first = manifest.read_bytes()
    release.assemble(args)
    assert manifest.read_bytes() == first
    release.validate(
        argparse.Namespace(
            manifest=manifest, tier=None, component=None, require_staged=False
        )
    )
    args.audit_output = args.output / "private"
    with pytest.raises(ValueError, match="outside the dataset"):
        release.assemble(args)


def gzip_text(path):
    import gzip

    with gzip.open(path, "rt") as handle:
        return handle.read()


def test_dataset_card_only_advertises_prepared_metadata(tmp_path):
    import yaml

    template = tmp_path / "card.md"
    template.write_text(
        "---\nlicense: mit\nconfigs:\n- config_name: files\n---\n\nApproved prose.\n"
    )
    root = tmp_path / "dataset"
    (root / "catalog/human_plays").mkdir(parents=True)
    (root / "catalog/human_plays/human_plays.parquet").touch()
    release.write_dataset_card(template, root)
    output = (root / "README.md").read_text()
    metadata = yaml.safe_load(output.split("---", 2)[1])
    assert [x["config_name"] for x in metadata["configs"]] == [
        "planned_files",
        "human_plays",
    ]
    assert metadata["configs"][0]["default"] is True
    assert all(c["data_files"][0]["split"] == "data" for c in metadata["configs"])
    assert output.endswith("\n\nApproved prose.\n")


def test_dataset_map_rejects_ambiguous_models_and_unsafe_payloads(row):
    row["metadata"]["source_model_id"] = "qwen35_27b"
    mapping = {
        "release_path": "features/lrm/qwen3.5-27b/input.pt",
        "selection_reason": "Reported representation",
    }
    with pytest.raises(ValueError, match="explicit public model_id"):
        release.dataset_source_row(row, mapping, "release")
    mapping["metadata"] = {"model_id": "qwen3.5-27b", "condition": "oracle"}
    result = release.dataset_source_row(row, mapping, "release")
    assert result["metadata"]["model_id"] == "qwen3.5-27b"
    assert result["metadata"]["source_model_id"] == "qwen35_27b"
    for path in (
        "behavior/data.bson",
        "behavior/export.imputed.replay.json.gz",
        "features/model%7Ccondition/a.pt",
    ):
        result["release_path"] = path
        with pytest.raises(
            ValueError, match="public dataset payload|encoded condition"
        ):
            release.check_public_dataset_paths([result])


def test_cached_source_requires_generation_and_byte_evidence(tmp_path, row):
    source = release.pin_source(row, Client())
    mapped = release.dataset_source_row(
        source,
        {"release_path": "features/ddqn/example.pt", "selection_reason": "fixture"},
        "v2",
    )
    local = tmp_path / "weights.pt"
    local.write_bytes(b"hello")
    receipt = {
        "local_path": str(local),
        "source_uri": row["source_uri"],
        "payload": {"sha256": hashlib.sha256(b"hello").hexdigest()},
    }
    with pytest.raises(ValueError, match="source-generation"):
        release.verify_local_source(receipt, mapped)
    receipt["source"] = source["source"]
    _, verified = release.verify_local_source(receipt, mapped)
    assert verified["artifact_id"] == row["artifact_id"]
    assert verified["payload"]["validation"] == "bytes-verified"
    local.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Cached bytes"):
        release.verify_local_source(receipt, verified)


def test_download_evidence_does_not_claim_local_or_published_payload(tmp_path, row):
    source_row = release.pin_source(row, Client())
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(source_row))
    mapping = tmp_path / "map.jsonl"
    destination = "features/ddqn/subject/game.npz"
    mapping.write_text(
        json.dumps(
            {
                "artifact_id": row["artifact_id"],
                "action": "include",
                "release_path": destination,
                "selection_reason": "Selected input",
            }
        )
    )
    evidence = {
        "source_uri": row["source_uri"],
        "source": source_row["source"],
        "payload": {
            "size_bytes": 5,
            "sha256": hashlib.sha256(b"hello").hexdigest(),
            "validation": "bytes-verified",
        },
    }
    enrichment = tmp_path / "evidence.jsonl"
    enrichment.write_text(json.dumps(evidence))
    args = argparse.Namespace(
        source_manifest=source,
        path_map=[mapping],
        local_files=[],
        human_manifest=None,
        human_directory=None,
        enrichments=[enrichment],
        release_id="test-v2",
        output=tmp_path / "dataset",
        audit_output=tmp_path / "private",
        workers=2,
        max_bytes=1000,
    )
    release.assemble(args)
    result = next(release.read_rows(args.output / "manifest.jsonl.gz"))
    assert result["payload"] == evidence["payload"]
    assert result["published"] is False
    assert result["artifact_id"] == row["artifact_id"]
    assert not (args.output / destination).exists()
    assert not list(release.read_rows(args.audit_output / "local-staging.jsonl.gz"))
    for changes, message in (
        (
            {"source": {**evidence["source"], "version_id": "different"}},
            "source generation",
        ),
        ({"payload": {**evidence["payload"], "size_bytes": 6}}, "verified size"),
        (
            {"payload": {**evidence["payload"], "validation": "headers-only"}},
            "verified size",
        ),
        ({"payload": {**evidence["payload"], "sha256": "bad"}}, "verified size"),
        (
            {"payload": {**evidence["payload"], "sha256": "a" * 64}},
            "checksum conflicts",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            release.apply_payload_evidence(
                {**evidence, **changes}, copy.deepcopy(result)
            )


def test_assembly_rejects_stale_bytes_at_reused_source_destination(tmp_path, row):
    source_row = release.pin_source(row, Client())
    source = tmp_path / "source.jsonl"
    mapping = tmp_path / "map.jsonl"
    receipt = tmp_path / "receipt.jsonl"
    cached = tmp_path / "cache.npz"
    destination = "features/ddqn/input.npz"

    def write_inputs(payload):
        source_row["payload"].update(
            sha256=hashlib.sha256(payload).hexdigest(), validation="bytes-verified"
        )
        source.write_text(json.dumps(source_row))
        mapping.write_text(
            json.dumps(
                {
                    "artifact_id": source_row["artifact_id"],
                    "action": "include",
                    "release_path": destination,
                    "selection_reason": "Selected input",
                }
            )
        )
        cached.write_bytes(payload)
        receipt.write_text(
            json.dumps(
                {
                    "source_uri": source_row["source_uri"],
                    "source": source_row["source"],
                    "payload": source_row["payload"],
                    "local_path": str(cached),
                }
            )
        )

    write_inputs(b"hello")
    args = argparse.Namespace(
        source_manifest=source,
        path_map=[mapping],
        local_files=[],
        source_files=[receipt],
        human_manifest=None,
        human_directory=None,
        enrichments=[],
        release_id="v1",
        output=tmp_path / "dataset",
        audit_output=tmp_path / "private",
        workers=2,
        max_bytes=100,
    )
    release.assemble(args)
    previous = (args.output / "manifest.jsonl.gz").read_bytes()
    source_row["source"]["version_id"] = "new-version"
    write_inputs(b"world")
    args.source_files = []
    with pytest.raises(ValueError, match="Existing payload does not match"):
        release.assemble(args)
    assert (args.output / "manifest.jsonl.gz").read_bytes() == previous
    assert (args.output / destination).read_bytes() == b"hello"
    source_row["payload"].update(sha256=None, validation="not-staged")
    source.write_text(json.dumps(source_row))
    with pytest.raises(ValueError, match="Existing payload does not match"):
        release.assemble(args)
    write_inputs(b"world")
    args.source_files = [receipt]
    release.assemble(args)
    result = next(release.read_rows(args.output / "manifest.jsonl.gz"))
    assert result["source"]["version_id"] == "new-version"
    assert (args.output / destination).read_bytes() == b"world"
    assert result["payload"]["sha256"] == hashlib.sha256(b"world").hexdigest()
    args.source_files = []
    release.assemble(args)
    retained = list(release.read_rows(args.audit_output / "local-staging.jsonl.gz"))
    assert len(retained) == 1
    assert retained[0]["release_path"] == destination
    assert retained[0]["sha256"] == result["payload"]["sha256"]
