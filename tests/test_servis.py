"""Dikey dilim API'si için entegrasyon testleri (sahte okuyucu ile; model indirmez)."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from obspy import Stream, Trace, UTCDateTime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from servis.api import uygulama_olustur  # noqa: E402
from servis.okuyucu import Okuma  # noqa: E402

T0 = UTCDateTime("2025-04-23T09:49:00")


class SahteOkuyucu:
    mimari, agirlik, esik = "Sahte", "test", 0.3

    def oku(self, akis):
        return [Okuma("TEST", "P", T0 + 10, 0.92), Okuma("TEST", "S", T0 + 17, 0.45)]


def mseed(istasyon="TEST", n_bilesen=3) -> bytes:
    st = Stream([Trace(np.random.default_rng(0).normal(size=3000).astype("float32"),
                       header={"network": "KO", "station": istasyon, "channel": f"HH{c}",
                               "sampling_rate": 100, "starttime": T0}) for c in "ZNE"[:n_bilesen]])
    b = io.BytesIO()
    st.write(b, format="MSEED")
    return b.getvalue()


@pytest.fixture
def istemci(tmp_path):
    return TestClient(uygulama_olustur(tmp_path / "t.db", SahteOkuyucu(), tmp_path / "arsiv"))


def test_pick_esige_gore_kuyruga_ayirir(istemci):
    r = istemci.post("/pick", files={"dosya": ("a.mseed", mseed())})
    assert r.status_code == 200
    durum = {o["faz"]: o["durum"] for o in r.json()["okumalar"]}
    assert durum == {"P": "otomatik_kabul", "S": "bekliyor"}
    kuyruk = istemci.get("/okumalar", params={"durum": "bekliyor"}).json()
    assert [o["faz"] for o in kuyruk] == ["S"]


def test_duzeltme_egitim_etiketine_doner(istemci):
    istemci.post("/pick", files={"dosya": ("a.mseed", mseed())})
    oid = istemci.get("/okumalar", params={"durum": "bekliyor"}).json()[0]["okuma_id"]
    yeni = str(T0 + 16.6)
    r = istemci.post(f"/okumalar/{oid}/inceleme", json={"analist": "a1", "karar": "duzelt", "duzeltilmis_zaman": yeni})
    assert r.json()["durum"] == "duzeltildi"
    etiket = istemci.get("/egitim-etiketleri").json()
    assert len(etiket) == 1 and etiket[0]["zaman"] == yeni and etiket[0]["analist"] == "a1"
    assert istemci.get("/okumalar", params={"durum": "bekliyor"}).json() == []


def test_reddedilen_okuma_etiket_olmaz(istemci):
    istemci.post("/pick", files={"dosya": ("a.mseed", mseed())})
    oid = istemci.get("/okumalar", params={"durum": "bekliyor"}).json()[0]["okuma_id"]
    istemci.post(f"/okumalar/{oid}/inceleme", json={"analist": "a1", "karar": "reddet"})
    assert istemci.get("/egitim-etiketleri").json() == []


def test_hatali_girdiler_reddedilir(istemci):
    assert istemci.post("/pick", files={"dosya": ("a.mseed", b"bozuk")}).status_code == 400
    assert istemci.post("/pick", files={"dosya": ("a.mseed", mseed(n_bilesen=1))}).status_code == 422
    assert istemci.post("/okumalar/999/inceleme", json={"analist": "a", "karar": "onayla"}).status_code == 404
    istemci.post("/pick", files={"dosya": ("a.mseed", mseed())})
    oid = istemci.get("/okumalar").json()[0]["okuma_id"]
    assert istemci.post(f"/okumalar/{oid}/inceleme", json={"analist": "a", "karar": "duzelt"}).status_code == 422
