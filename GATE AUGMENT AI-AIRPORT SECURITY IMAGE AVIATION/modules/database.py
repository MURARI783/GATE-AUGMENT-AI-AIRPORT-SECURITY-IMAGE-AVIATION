from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


@contextmanager
def _connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            processing_ms REAL NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            operation TEXT NOT NULL,
            result_path TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            event_type TEXT NOT NULL,
            description TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );
        """
    )
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def log_event(db_path: Path, event_type: str, description: str, metadata: dict | None = None) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            "INSERT INTO events(created_at,event_type,description,metadata_json) VALUES(?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                event_type,
                description,
                json.dumps(metadata or {}, allow_nan=False),
            ),
        )


def record_scan(db_path: Path, scan: dict) -> int:
    with _connect(db_path) as connection:
        cursor = connection.execute(
            """INSERT INTO scans
            (created_at,source,filename,status,processing_ms,width,height,operation,result_path,metadata_json)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                scan.get("created_at", datetime.now(timezone.utc).isoformat()),
                scan["source"],
                scan["filename"],
                scan["status"],
                float(scan["processing_ms"]),
                int(scan["width"]),
                int(scan["height"]),
                scan["operation"],
                scan["result_path"],
                json.dumps(scan.get("metadata", {}), allow_nan=False),
            ),
        )
        return int(cursor.lastrowid)


def recent_scans(db_path: Path, limit: int = 50) -> list[dict]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ?", (max(1, min(int(limit), 500)),)
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json"))
        result.append(item)
    return result


def event_summary(db_path: Path) -> dict:
    with _connect(db_path) as connection:
        scan_row = connection.execute(
            """SELECT COUNT(*) AS count,
            SUM(CASE WHEN source='video' THEN 1 ELSE 0 END) AS video_frames,
            SUM(CASE WHEN source='webcam_demo' THEN 1 ELSE 0 END) AS webcam_frames
            FROM scans"""
        ).fetchone()
        events = connection.execute(
            "SELECT event_type, COUNT(*) AS count FROM events GROUP BY event_type"
        ).fetchall()
    return {
        "scans": int(scan_row["count"] or 0),
        "video_frames": int(scan_row["video_frames"] or 0),
        "webcam_frames": int(scan_row["webcam_frames"] or 0),
        "events": {row["event_type"]: int(row["count"]) for row in events},
    }


def recent_events(db_path: Path, limit: int = 30) -> list[dict]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (max(1, min(int(limit), 500)),)
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json"))
        result.append(item)
    return result
