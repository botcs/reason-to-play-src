"""DynamoDB client for the experiment grid tracker."""

import gzip
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

BACKUP_DIR = "out/db_backups"


def _decimal_default(obj):
    """JSON serializer for Decimal values returned by DynamoDB."""
    if isinstance(obj, Decimal):
        return int(obj) if obj == int(obj) else float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# Table name constants
TABLE_GAMEPLAY = "vgdl-grid-gameplay"
TABLE_REPLAY = "vgdl-grid-replay"
TABLE_FEATURE_EXTRACTION = "vgdl-grid-feature-extraction"
TABLE_ABLATION = "vgdl-grid-ablation"
TABLE_BEHAVIOURAL = "vgdl-behavioural-summary"

ALL_TABLES = [
    TABLE_GAMEPLAY,
    TABLE_REPLAY,
    TABLE_FEATURE_EXTRACTION,
    TABLE_ABLATION,
    TABLE_BEHAVIOURAL,
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExpDB:
    """Thin wrapper around DynamoDB for experiment slot CRUD."""

    def __init__(self, region: str = "eu-west-2"):
        self._dynamo = boto3.resource("dynamodb", region_name=region)
        self._s3 = boto3.client("s3", region_name=region)
        self._table_cache: dict[str, object] = {}

    def _table(self, name: str):
        if name not in self._table_cache:
            self._table_cache[name] = self._dynamo.Table(name)
        return self._table_cache[name]

    # -- Create ----------------------------------------------------------------

    def create_slot(self, table: str, slot_id: str, attributes: dict) -> bool:
        """Insert a slot if it does not already exist (idempotent).

        Returns True if created, False if it already existed.
        """
        item = {"slot_id": slot_id, "status": "pending", **attributes}
        t = self._table(table)
        try:
            t.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(slot_id)",
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise
        return True

    def batch_create_slots(self, table: str, items: list[dict]) -> int:
        """UNCONDITIONAL batch put -- DESTRUCTIVE on existing slots.

        DynamoDB's batch_writer does not support ConditionExpression, so
        every put_item here unconditionally OVERWRITES any existing item
        with the same slot_id (status=pending, no wandb_run_id, no
        finished_at, etc. -- the new item shape replaces the old item
        completely).

        Do NOT use this on a populated table.  For idempotent population
        with diff semantics use ``populate.populate_from_yaml`` (it calls
        ``create_slot`` -- conditional on ``attribute_not_exists`` -- and
        skips slots that already exist).

        Uses batch_write_item (25 items per request) for throughput.
        Returns the number of items written (= len(items)).
        """
        t = self._table(table)
        written = 0
        for i in range(0, len(items), 25):
            batch = items[i : i + 25]
            with t.batch_writer() as writer:
                for item in batch:
                    writer.put_item(Item=item)
            written += len(batch)
        return written

    # -- Claim / update --------------------------------------------------------

    def claim_slot(
        self,
        table: str,
        slot_id: str,
        worker: str,
        wandb_run_id: str,
        wandb_run_url: str,
        s3_replay_key: str | None = None,
    ) -> bool:
        """Atomically transition a slot from pending -> running.

        Returns True if claimed, False if slot was not in pending status.
        """
        t = self._table(table)
        expr = (
            "SET #s = :running, started_at = :now, worker = :w,"
            " wandb_run_id = :wid, wandb_run_url = :wurl"
        )
        values = {
            ":running": "running",
            ":pending": "pending",
            ":now": _now_iso(),
            ":w": worker,
            ":wid": wandb_run_id,
            ":wurl": wandb_run_url,
        }
        if s3_replay_key is not None:
            expr += ", s3_replay_key = :s3k"
            values[":s3k"] = s3_replay_key
        try:
            t.update_item(
                Key={"slot_id": slot_id},
                UpdateExpression=expr,
                ConditionExpression="#s = :pending",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues=values,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise
        return True

    def reclaim_slot(
        self,
        table: str,
        slot_id: str,
        worker: str,
        wandb_run_id: str,
        wandb_run_url: str,
        s3_replay_key: str | None = None,
    ) -> bool:
        """Atomically transition a slot from failed -> running (for resume).

        Returns True if reclaimed, False if slot was not in failed status.
        """
        t = self._table(table)
        expr = (
            "SET #s = :running, started_at = :now, worker = :w,"
            " wandb_run_id = :wid, wandb_run_url = :wurl"
        )
        values = {
            ":running": "running",
            ":failed": "failed",
            ":now": _now_iso(),
            ":w": worker,
            ":wid": wandb_run_id,
            ":wurl": wandb_run_url,
        }
        if s3_replay_key is not None:
            expr += ", s3_replay_key = :s3k"
            values[":s3k"] = s3_replay_key
        try:
            t.update_item(
                Key={"slot_id": slot_id},
                UpdateExpression=expr,
                ConditionExpression="#s = :failed",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues=values,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise
        return True

    def download_replay(self, bucket: str, key: str, local_path: str | Path) -> Path:
        """Download a replay .json.gz from S3."""
        local_path = Path(local_path)
        self._s3.download_file(bucket, key, str(local_path))
        return local_path

    def upload_replay(self, bucket: str, key: str, local_path: str | Path) -> str:
        """Upload a replay file to S3.

        Returns the s3:// URI of the uploaded object.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Replay file not found: {local_path}")
        self._s3.upload_file(str(local_path), bucket, key)
        return f"s3://{bucket}/{key}"

    def complete_slot(
        self,
        table: str,
        slot_id: str,
        s3_replay_key: str | None = None,
    ) -> None:
        """Mark a slot as done. wandb fields are already set by claim_slot."""
        expr = "SET #s = :done, finished_at = :now"
        values: dict = {":done": "done", ":now": _now_iso()}
        if s3_replay_key is not None:
            expr += ", s3_replay_key = :s3k"
            values[":s3k"] = s3_replay_key

        self._table(table).update_item(
            Key={"slot_id": slot_id},
            UpdateExpression=expr,
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues=values,
        )

    def fail_slot(self, table: str, slot_id: str, error_msg: str | None = None) -> None:
        """Mark a slot as failed."""
        expr = "SET #s = :failed, finished_at = :now"
        values: dict = {":failed": "failed", ":now": _now_iso()}
        if error_msg is not None:
            expr += ", #e = :emsg"
            values[":emsg"] = error_msg[:2000]  # DynamoDB item size guard

        names = {"#s": "status"}
        if error_msg is not None:
            names["#e"] = "error"

        self._table(table).update_item(
            Key={"slot_id": slot_id},
            UpdateExpression=expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )

    def set_status(
        self, table: str, slot_id: str, status: str, notes: str | None = None
    ) -> None:
        """Unconditional status update (for manual overrides)."""
        expr = "SET #s = :st"
        values: dict = {":st": status}
        names: dict = {"#s": "status"}
        if notes is not None:
            expr += ", notes = :n"
            values[":n"] = notes
        if status == "pending":
            # Clear runtime fields on reset
            expr += " REMOVE started_at, finished_at, worker, #e, wandb_run_id"
            names["#e"] = "error"

        self._table(table).update_item(
            Key={"slot_id": slot_id},
            UpdateExpression=expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
        )

    # -- Query -----------------------------------------------------------------

    def get_slot(self, table: str, slot_id: str) -> dict | None:
        """Fetch a single slot by ID."""
        resp = self._table(table).get_item(Key={"slot_id": slot_id})
        return resp.get("Item")

    def batch_get_slots(self, table: str, slot_ids: list[str]) -> dict[str, dict]:
        """Fetch many slots by ID in 100-key chunks via BatchGetItem.

        Returns a dict mapping ``slot_id`` -> item.  Missing slots are
        absent from the result.  Handles ``UnprocessedKeys`` retries when
        DynamoDB throttles a chunk.  Use this instead of N point-gets when
        checking grid status -- one batch call is dramatically cheaper than
        100 round-trips, especially cross-region.
        """
        unique_ids = list({sid for sid in slot_ids if sid})
        if not unique_ids:
            return {}
        client = self._table(table).meta.client
        out: dict[str, dict] = {}
        for i in range(0, len(unique_ids), 100):
            chunk = unique_ids[i : i + 100]
            request = {table: {"Keys": [{"slot_id": s} for s in chunk]}}
            while request:
                resp = client.batch_get_item(RequestItems=request)
                for item in resp.get("Responses", {}).get(table, []):
                    out[item["slot_id"]] = item
                request = resp.get("UnprocessedKeys") or None
        return out

    def query_by_status(self, table: str, status: str) -> list[dict]:
        """Query slots by status via GSI."""
        resp = self._table(table).query(
            IndexName="status-index",
            KeyConditionExpression="#s = :st",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":st": status},
        )
        items = resp["Items"]
        while "LastEvaluatedKey" in resp:
            resp = self._table(table).query(
                IndexName="status-index",
                KeyConditionExpression="#s = :st",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={":st": status},
                ExclusiveStartKey=resp["LastEvaluatedKey"],
            )
            items.extend(resp["Items"])
        return items

    def query_by_model(self, table: str, model: str) -> list[dict]:
        """Query slots by model via GSI."""
        resp = self._table(table).query(
            IndexName="model-index",
            KeyConditionExpression="model = :m",
            ExpressionAttributeValues={":m": model},
        )
        items = resp["Items"]
        while "LastEvaluatedKey" in resp:
            resp = self._table(table).query(
                IndexName="model-index",
                KeyConditionExpression="model = :m",
                ExpressionAttributeValues={":m": model},
                ExclusiveStartKey=resp["LastEvaluatedKey"],
            )
            items.extend(resp["Items"])
        return items

    def scan_all(self, table: str) -> list[dict]:
        """Full table scan. Fine for <100k items."""
        resp = self._table(table).scan()
        items = resp["Items"]
        while "LastEvaluatedKey" in resp:
            resp = self._table(table).scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
            items.extend(resp["Items"])
        return items

    def backup_table(
        self, table: str, backup_dir: str = BACKUP_DIR, tag: str = ""
    ) -> Path | None:
        """Snapshot a table to a compressed JSON file.

        Returns the backup path, or ``None`` if the table was empty.
        Every script that mutates experiment-DB state should call this
        before making changes so the operation is reversible.
        """
        items = self.scan_all(table)
        if not items:
            print(f"  backup {table}: empty, nothing to back up")
            return None
        out = Path(backup_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"_{tag}" if tag else ""
        fname = out / f"{table}{suffix}_{ts}.json.gz"
        with gzip.open(fname, "wt", encoding="utf-8") as f:
            json.dump(items, f, default=_decimal_default)
        print(f"  backup {table}: {len(items)} items -> {fname}")
        return fname

    def purge_table(self, table: str) -> int:
        """Delete all items from a table. Returns the number of items deleted."""
        items = self.scan_all(table)
        t = self._table(table)
        deleted = 0
        for i in range(0, len(items), 25):
            batch = items[i : i + 25]
            with t.batch_writer() as writer:
                for item in batch:
                    writer.delete_item(Key={"slot_id": item["slot_id"]})
            deleted += len(batch)
        return deleted

    def summary(self, table: str) -> dict[str, int]:
        """Return status counts for a table."""
        items = self.scan_all(table)
        counts: dict[str, int] = {}
        for item in items:
            s = item.get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return counts
