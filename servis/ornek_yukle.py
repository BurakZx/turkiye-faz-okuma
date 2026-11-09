"""Ön deneme kayıtlarından birkaçını API'ye yükler (panelde gösterim için).

Çalıştırma:  .venv/bin/python -m servis.ornek_yukle [olay_kimligi] [en_fazla_kayit]
"""
import sys
from pathlib import Path

import httpx

API = "http://127.0.0.1:8000"
VERI = Path(__file__).resolve().parents[1] / "veri" / "mseed"


def main(evid: str = "us7000pufs", en_fazla: int = 12):
    for dosya in sorted(VERI.glob(f"{evid}_*.mseed"))[:en_fazla]:
        if dosya.stat().st_size == 0:
            continue
        r = httpx.post(f"{API}/pick", files={"dosya": (dosya.name, dosya.read_bytes())}, timeout=120)
        r.raise_for_status()
        j = r.json()
        print(j["istasyon"], [(o["faz"], o["olasilik"], o["durum"]) for o in j["okumalar"]])


if __name__ == "__main__":
    main(*(sys.argv[1:2]), *([int(sys.argv[2])] if len(sys.argv) > 2 else []))
