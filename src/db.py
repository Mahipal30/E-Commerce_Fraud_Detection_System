import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path("data") / "app.db"

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS scored_transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        fraud_proba REAL NOT NULL,
        risk_score INTEGER NOT NULL,
        risk_band TEXT NOT NULL
    )
    """)
    return conn

def insert_score(payload: dict, fraud_proba: float, risk_score: int, risk_band: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO scored_transactions (created_at, payload_json, fraud_proba, risk_score, risk_band) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), json.dumps(payload), float(fraud_proba), int(risk_score), str(risk_band))
    )
    conn.commit()
    conn.close()

def fetch_recent(limit: int = 50):
    conn = get_conn()
    cur = conn.execute(
        "SELECT created_at, payload_json, fraud_proba, risk_score, risk_band FROM scored_transactions ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    return rows