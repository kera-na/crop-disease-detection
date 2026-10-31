"""Storage, geocoding and distance helpers for the agriculture expert directory."""

import json
import math
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DB_PATH = os.getenv("EXPERT_DB_PATH", "data/experts.db")

SPECIALIZATIONS = [
    "Agricultural extension officer",
    "Plant pathologist (disease specialist)",
    "Agronomist / crop consultant",
    "Soil and fertilizer specialist",
    "Pest management specialist",
    "Agricultural research officer",
]


def _connect():
    folder = os.path.dirname(DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                specialization TEXT NOT NULL,
                organisation TEXT,
                crops TEXT,
                phone TEXT NOT NULL,
                email TEXT,
                location TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                notes TEXT,
                verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )


def add_expert(name, specialization, organisation, crops, phone, email,
               location, lat, lon, notes):
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            INSERT INTO experts
              (name, specialization, organisation, crops, phone, email,
               location, lat, lon, notes, verified, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (name, specialization, organisation, ",".join(crops), phone, email,
             location, float(lat), float(lon), notes,
             datetime.now().strftime("%Y-%m-%d %H:%M")),
        )


def list_experts():
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT * FROM experts ORDER BY verified DESC, name").fetchall()
    return [dict(r) for r in rows]


def set_verified(expert_id, verified):
    with closing(_connect()) as conn, conn:
        conn.execute("UPDATE experts SET verified = ? WHERE id = ?", (int(verified), expert_id))


def delete_expert(expert_id):
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM experts WHERE id = ?", (expert_id,))


def geocode(query):
    """Look up a place name with OpenStreetMap Nominatim. Returns (lat, lon, label) or None."""
    query = (query or "").strip()
    if not query:
        return None
    url = "https://nominatim.openstreetmap.org/search?" + urlencode(
        {"q": query, "format": "json", "limit": 1}
    )
    request = Request(url, headers={"User-Agent": "CropDiseaseDetectionSystem/1.0"})
    try:
        with urlopen(request, timeout=8) as response:
            data = json.load(response)
    except Exception:
        return None
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", query)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def db_info():
    """Basic storage details for the admin panel."""
    experts = list_experts()
    exists = os.path.exists(DB_PATH)
    return {
        "path": os.path.abspath(DB_PATH),
        "size_kb": round(os.path.getsize(DB_PATH) / 1024, 1) if exists else 0.0,
        "total": len(experts),
        "verified": sum(1 for e in experts if e["verified"]),
    }


def import_experts(records):
    """Restore experts from a backup list. Skips entries that already exist (same name and phone)."""
    required = ("name", "specialization", "phone", "location", "lat", "lon")
    existing = {(e["name"].strip().lower(), e["phone"].strip()) for e in list_experts()}
    added = skipped = 0
    with closing(_connect()) as conn, conn:
        for r in records:
            if not isinstance(r, dict) or any(r.get(k) in (None, "") for k in required):
                skipped += 1
                continue
            key = (str(r["name"]).strip().lower(), str(r["phone"]).strip())
            if key in existing:
                skipped += 1
                continue
            conn.execute(
                """
                INSERT INTO experts
                  (name, specialization, organisation, crops, phone, email,
                   location, lat, lon, notes, verified, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(r["name"]), str(r["specialization"]), r.get("organisation") or "",
                    r.get("crops") or "", str(r["phone"]), r.get("email") or "",
                    str(r["location"]), float(r["lat"]), float(r["lon"]),
                    r.get("notes") or "", int(bool(r.get("verified"))),
                    r.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M"),
                ),
            )
            existing.add(key)
            added += 1
    return added, skipped