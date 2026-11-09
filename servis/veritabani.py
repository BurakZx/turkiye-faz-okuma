"""Dikey dilim için SQLite şeması (ER diyagramının sadeleştirilmiş alt kümesi).

Asıl sistemde PostgreSQL kullanılacaktır; tablo ve alan adları ER diyagramıyla aynı tutulmuştur.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SEMA = """
CREATE TABLE IF NOT EXISTS model_surumu (
    model_surumu_id INTEGER PRIMARY KEY,
    mimari TEXT NOT NULL, agirlik TEXT NOT NULL, esik REAL NOT NULL,
    UNIQUE (mimari, agirlik, esik)
);
CREATE TABLE IF NOT EXISTS kayit (
    kayit_id INTEGER PRIMARY KEY,
    ag TEXT, istasyon TEXT NOT NULL, baslangic TEXT NOT NULL, bitis TEXT NOT NULL,
    dosya TEXT NOT NULL, yuklenme TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS faz_okumasi (
    okuma_id INTEGER PRIMARY KEY,
    kayit_id INTEGER NOT NULL REFERENCES kayit(kayit_id),
    model_surumu_id INTEGER REFERENCES model_surumu(model_surumu_id),
    faz TEXT NOT NULL CHECK (faz IN ('P', 'S')),
    zaman TEXT NOT NULL, olasilik REAL NOT NULL,
    kaynak TEXT NOT NULL DEFAULT 'MODEL',
    durum TEXT NOT NULL CHECK (durum IN ('otomatik_kabul', 'bekliyor', 'onaylandi', 'duzeltildi', 'reddedildi'))
);
CREATE TABLE IF NOT EXISTS analist_incelemesi (
    inceleme_id INTEGER PRIMARY KEY,
    okuma_id INTEGER NOT NULL REFERENCES faz_okumasi(okuma_id),
    analist TEXT NOT NULL,
    karar TEXT NOT NULL CHECK (karar IN ('onayla', 'duzelt', 'reddet')),
    duzeltilmis_zaman TEXT,
    zaman TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


def baglan(yol: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(yol), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SEMA)
    return con
