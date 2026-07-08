"""SQLite 저장소: 프로젝트/용어집/참석자, 화자 보이스프린트, 작업 큐, 회의 기록."""
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np

from .config import DB_PATH, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects(
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  description TEXT DEFAULT '',
  minutes_template TEXT DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS glossary(
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  term TEXT NOT NULL,
  note TEXT DEFAULT '',
  UNIQUE(project_id, term)
);
CREATE TABLE IF NOT EXISTS participants(
  id INTEGER PRIMARY KEY,
  project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  role TEXT DEFAULT '',
  UNIQUE(project_id, name)
);
CREATE TABLE IF NOT EXISTS speakers(
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  consent_note TEXT DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS speaker_embeddings(
  id INTEGER PRIMARY KEY,
  speaker_id INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
  embedding BLOB NOT NULL,
  dim INTEGER NOT NULL,
  source TEXT DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  input_path TEXT NOT NULL,
  project_id INTEGER,
  status TEXT NOT NULL,
  stage TEXT DEFAULT '',
  error TEXT DEFAULT '',
  options TEXT DEFAULT '{}',
  result_dir TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_speaker_map(
  job_id TEXT NOT NULL,
  cluster_label TEXT NOT NULL,
  suggested_speaker_id INTEGER,
  similarity REAL,
  confirmed_speaker_id INTEGER,
  embedding BLOB,
  dim INTEGER,
  PRIMARY KEY(job_id, cluster_label)
);
CREATE TABLE IF NOT EXISTS meetings(
  id INTEGER PRIMARY KEY,
  project_id INTEGER REFERENCES projects(id),
  job_id TEXT,
  title TEXT DEFAULT '',
  meeting_date TEXT DEFAULT '',
  minutes_md TEXT DEFAULT '',
  summary TEXT DEFAULT '',
  created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_conn() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ---------- 프로젝트 / 컨텍스트 ----------

def upsert_project(name: str, description: str = "", minutes_template: str = "") -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM projects WHERE name=?", (name,)).fetchone()
        if row:
            sets, args = [], []
            if description:
                sets.append("description=?"); args.append(description)
            if minutes_template:
                sets.append("minutes_template=?"); args.append(minutes_template)
            if sets:
                args.append(row["id"])
                conn.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id=?", args)
            return row["id"]
        cur = conn.execute(
            "INSERT INTO projects(name, description, minutes_template, created_at) VALUES(?,?,?,?)",
            (name, description, minutes_template, now()),
        )
        return cur.lastrowid


def get_project(name: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM projects WHERE name=?", (name,)).fetchone()


def list_projects() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT p.name, p.description,
                      (SELECT COUNT(*) FROM glossary g WHERE g.project_id=p.id) AS terms,
                      (SELECT COUNT(*) FROM participants pa WHERE pa.project_id=p.id) AS participants,
                      (SELECT COUNT(*) FROM meetings m WHERE m.project_id=p.id) AS meetings
               FROM projects p ORDER BY p.id"""
        ).fetchall()
        return [dict(r) for r in rows]


def add_glossary_terms(project_id: int, terms: list[dict | str]) -> int:
    added = 0
    with get_conn() as conn:
        for t in terms:
            term, note = (t, "") if isinstance(t, str) else (t.get("term", ""), t.get("note", ""))
            term = term.strip()
            if not term:
                continue
            cur = conn.execute(
                "INSERT INTO glossary(project_id, term, note) VALUES(?,?,?) "
                "ON CONFLICT(project_id, term) DO UPDATE SET note=excluded.note WHERE excluded.note != ''",
                (project_id, term, note),
            )
            added += cur.rowcount
    return added


def add_participants(project_id: int, people: list[dict | str]) -> int:
    added = 0
    with get_conn() as conn:
        for p in people:
            name, role = (p, "") if isinstance(p, str) else (p.get("name", ""), p.get("role", ""))
            name = name.strip()
            if not name:
                continue
            cur = conn.execute(
                "INSERT INTO participants(project_id, name, role) VALUES(?,?,?) "
                "ON CONFLICT(project_id, name) DO UPDATE SET role=excluded.role WHERE excluded.role != ''",
                (project_id, name, role),
            )
            added += cur.rowcount
    return added


def get_context(project_id: int, recent_meetings: int = 5) -> dict:
    with get_conn() as conn:
        proj = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        glossary = conn.execute(
            "SELECT term, note FROM glossary WHERE project_id=? ORDER BY id", (project_id,)
        ).fetchall()
        people = conn.execute(
            "SELECT name, role FROM participants WHERE project_id=? ORDER BY id", (project_id,)
        ).fetchall()
        meetings = conn.execute(
            "SELECT title, meeting_date, summary FROM meetings WHERE project_id=? "
            "ORDER BY id DESC LIMIT ?",
            (project_id, recent_meetings),
        ).fetchall()
    return {
        "name": proj["name"],
        "description": proj["description"],
        "minutes_template": proj["minutes_template"],
        "glossary": [dict(g) for g in glossary],
        "participants": [dict(p) for p in people],
        "recent_meetings": [dict(m) for m in meetings],
    }


def hotwords_for_project(project_id: int, limit: int) -> list[str]:
    """ASR 부스팅용 핵심 용어: 참석자 이름 우선 + 최신 용어 순."""
    with get_conn() as conn:
        names = [r["name"] for r in conn.execute(
            "SELECT name FROM participants WHERE project_id=? ORDER BY id", (project_id,))]
        terms = [r["term"] for r in conn.execute(
            "SELECT term FROM glossary WHERE project_id=? ORDER BY id DESC", (project_id,))]
    out: list[str] = []
    for w in names + terms:
        if w not in out:
            out.append(w)
        if len(out) >= limit:
            break
    return out


# ---------- 화자 ----------

def emb_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_emb(b: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32, count=dim)


def upsert_speaker(name: str, consent_note: str = "") -> int:
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM speakers WHERE name=?", (name,)).fetchone()
        if row:
            if consent_note:
                conn.execute("UPDATE speakers SET consent_note=? WHERE id=?", (consent_note, row["id"]))
            return row["id"]
        cur = conn.execute(
            "INSERT INTO speakers(name, consent_note, created_at) VALUES(?,?,?)",
            (name, consent_note, now()),
        )
        return cur.lastrowid


def add_speaker_embedding(speaker_id: int, emb: np.ndarray, source: str = "") -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO speaker_embeddings(speaker_id, embedding, dim, source, created_at) VALUES(?,?,?,?,?)",
            (speaker_id, emb_to_blob(emb), len(emb), source, now()),
        )


def all_speaker_embeddings() -> list[tuple[int, str, np.ndarray]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT s.id, s.name, e.embedding, e.dim FROM speakers s "
            "JOIN speaker_embeddings e ON e.speaker_id = s.id"
        ).fetchall()
    return [(r["id"], r["name"], blob_to_emb(r["embedding"], r["dim"])) for r in rows]


def list_speakers() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT s.name, s.consent_note, s.created_at, COUNT(e.id) AS embeddings "
            "FROM speakers s LEFT JOIN speaker_embeddings e ON e.speaker_id=s.id "
            "GROUP BY s.id ORDER BY s.id"
        ).fetchall()
        return [dict(r) for r in rows]


def purge_speaker(name: str) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM speakers WHERE name=?", (name,)).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM speakers WHERE id=?", (row["id"],))
        conn.execute(
            "UPDATE job_speaker_map SET suggested_speaker_id=NULL, similarity=NULL "
            "WHERE suggested_speaker_id=?", (row["id"],))
        conn.execute(
            "UPDATE job_speaker_map SET confirmed_speaker_id=NULL WHERE confirmed_speaker_id=?",
            (row["id"],))
        return True


# ---------- 작업 ----------

def create_job(job_id: str, kind: str, input_path: str, project_id: int | None,
               options: dict, result_dir: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO jobs(id, kind, input_path, project_id, status, options, result_dir, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (job_id, kind, input_path, project_id, "queued", json.dumps(options, ensure_ascii=False),
             result_dir, now(), now()),
        )


def update_job(job_id: str, **fields) -> None:
    sets = ", ".join(f"{k}=?" for k in fields)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE jobs SET {sets}, updated_at=? WHERE id=?",
            (*fields.values(), now(), job_id),
        )


def get_job(job_id: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()


def pending_jobs() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status IN ('queued','running') ORDER BY created_at"
        ).fetchall()
        return [r["id"] for r in rows]


def set_job_speaker_map(job_id: str, cluster_label: str, emb: np.ndarray | None,
                        suggested_speaker_id: int | None, similarity: float | None) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO job_speaker_map(job_id, cluster_label, suggested_speaker_id, similarity, embedding, dim) "
            "VALUES(?,?,?,?,?,?)",
            (job_id, cluster_label, suggested_speaker_id, similarity,
             emb_to_blob(emb) if emb is not None else None,
             len(emb) if emb is not None else None),
        )


def get_job_speaker_map(job_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT m.*, ss.name AS suggested_name, cs.name AS confirmed_name
               FROM job_speaker_map m
               LEFT JOIN speakers ss ON ss.id = m.suggested_speaker_id
               LEFT JOIN speakers cs ON cs.id = m.confirmed_speaker_id
               WHERE m.job_id=? ORDER BY m.cluster_label""",
            (job_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def confirm_cluster(job_id: str, cluster_label: str, speaker_id: int) -> np.ndarray | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT embedding, dim FROM job_speaker_map WHERE job_id=? AND cluster_label=?",
            (job_id, cluster_label),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE job_speaker_map SET confirmed_speaker_id=? WHERE job_id=? AND cluster_label=?",
            (speaker_id, job_id, cluster_label),
        )
        if row["embedding"] is not None:
            return blob_to_emb(row["embedding"], row["dim"])
        return None


# ---------- 회의 기록 ----------

def save_meeting(project_id: int | None, job_id: str, title: str, meeting_date: str,
                 minutes_md: str, summary: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO meetings(project_id, job_id, title, meeting_date, minutes_md, summary, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (project_id, job_id, title, meeting_date, minutes_md, summary, now()),
        )
        return cur.lastrowid
