"""Dikey dilim: faz okuma REST API'si (FastAPI + SQLite).

Uç noktalar
  GET  /saglik                       servis ayakta mı
  POST /pick                         MiniSEED yükle → model okur → okumalar kaydedilir
  GET  /okumalar?durum=bekliyor      inceleme kuyruğu (düşük olasılık önce)
  POST /okumalar/{id}/inceleme       analist kararı: onayla | duzelt | reddet
  GET  /egitim-etiketleri            onaylanan/düzeltilen okumalar (yeniden eğitim havuzu, İG-11)

Çalıştırma:  .venv/bin/uvicorn servis.api:app --port 8000
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from obspy import UTCDateTime, read
from pydantic import BaseModel

from .okuyucu import FazOkuyucu
from .veritabani import baglan

KOK = Path(__file__).resolve().parents[1]
OTOMATIK_KABUL_ESIGI = 0.7  # başlangıç değeri; kalibrasyon (ECE) sonrası güncellenecek


class Inceleme(BaseModel):
    analist: str
    karar: Literal["onayla", "duzelt", "reddet"]
    duzeltilmis_zaman: str | None = None


def uygulama_olustur(db_yolu: str | Path, okuyucu: FazOkuyucu, arsiv: str | Path) -> FastAPI:
    con = baglan(db_yolu)
    arsiv = Path(arsiv)
    arsiv.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Faz Okuma Servisi (dikey dilim)", version="0.1")
    con.execute("INSERT OR IGNORE INTO model_surumu (mimari, agirlik, esik) VALUES (?, ?, ?)",
                (okuyucu.mimari, okuyucu.agirlik, getattr(okuyucu, "esik", 0.3)))
    msid = con.execute("SELECT model_surumu_id FROM model_surumu WHERE mimari=? AND agirlik=?",
                       (okuyucu.mimari, okuyucu.agirlik)).fetchone()[0]
    con.commit()

    @app.get("/saglik")
    def saglik():
        return {"durum": "ok", "model": f"{okuyucu.mimari}-{okuyucu.agirlik}"}

    @app.post("/pick")
    def pick(dosya: UploadFile = File(...)):  # eşzamanlı: SeisBench kendi olay döngüsünü açar
        ham = dosya.file.read()
        try:
            akis = read(io.BytesIO(ham))
        except Exception as e:  # obspy okunamayan dosyada çeşitli istisnalar atar
            raise HTTPException(400, f"MiniSEED okunamadı: {e}")
        if len({tr.stats.station for tr in akis}) != 1 or len(akis) < 3:
            raise HTTPException(422, "Tek istasyonun üç bileşenli kaydı bekleniyor")
        akis.merge(fill_value=0)
        ist, ag = akis[0].stats.station, akis[0].stats.network
        bas, bit = min(tr.stats.starttime for tr in akis), max(tr.stats.endtime for tr in akis)
        yol = arsiv / f"{ag}.{ist}.{bas.strftime('%Y%m%dT%H%M%S')}.mseed"
        yol.write_bytes(ham)
        kid = con.execute("INSERT INTO kayit (ag, istasyon, baslangic, bitis, dosya) VALUES (?,?,?,?,?)",
                          (ag, ist, str(bas), str(bit), str(yol))).lastrowid
        cikti = []
        for o in okuyucu.oku(akis):
            durum = "otomatik_kabul" if o.olasilik >= OTOMATIK_KABUL_ESIGI else "bekliyor"
            oid = con.execute("INSERT INTO faz_okumasi (kayit_id, model_surumu_id, faz, zaman, olasilik, durum) "
                              "VALUES (?,?,?,?,?,?)", (kid, msid, o.faz, str(o.zaman), o.olasilik, durum)).lastrowid
            cikti.append({"okuma_id": oid, "faz": o.faz, "zaman": str(o.zaman),
                          "olasilik": round(o.olasilik, 3), "durum": durum})
        con.commit()
        return {"kayit_id": kid, "istasyon": f"{ag}.{ist}", "okumalar": cikti}

    @app.get("/okumalar")
    def okumalar(durum: str | None = Query(None)):
        sql = ("SELECT f.*, k.ag, k.istasyon, k.dosya FROM faz_okumasi f JOIN kayit k USING (kayit_id)"
               + (" WHERE f.durum = ?" if durum else "") + " ORDER BY f.olasilik ASC, f.okuma_id")
        return [dict(r) for r in con.execute(sql, (durum,) if durum else ())]

    @app.post("/okumalar/{okuma_id}/inceleme")
    def incele(okuma_id: int, i: Inceleme):
        r = con.execute("SELECT * FROM faz_okumasi WHERE okuma_id=?", (okuma_id,)).fetchone()
        if r is None:
            raise HTTPException(404, "Okuma bulunamadı")
        if i.karar == "duzelt":
            if not i.duzeltilmis_zaman:
                raise HTTPException(422, "Düzeltme için duzeltilmis_zaman gerekli")
            UTCDateTime(i.duzeltilmis_zaman)  # biçim doğrulaması
        con.execute("INSERT INTO analist_incelemesi (okuma_id, analist, karar, duzeltilmis_zaman) VALUES (?,?,?,?)",
                    (okuma_id, i.analist, i.karar, i.duzeltilmis_zaman))
        yeni = {"onayla": "onaylandi", "duzelt": "duzeltildi", "reddet": "reddedildi"}[i.karar]
        con.execute("UPDATE faz_okumasi SET durum=? WHERE okuma_id=?", (yeni, okuma_id))
        con.commit()
        return {"okuma_id": okuma_id, "durum": yeni}

    @app.get("/egitim-etiketleri")
    def egitim_etiketleri():
        """Analistin onayladığı ya da düzelttiği okumalar: yeniden eğitim için etiket havuzu."""
        sql = """SELECT k.ag, k.istasyon, k.dosya, f.faz,
                        COALESCE(a.duzeltilmis_zaman, f.zaman) AS zaman, a.karar, a.analist, a.zaman AS inceleme_zamani
                 FROM faz_okumasi f JOIN kayit k USING (kayit_id)
                 JOIN analist_incelemesi a ON a.okuma_id = f.okuma_id
                 WHERE a.karar IN ('onayla', 'duzelt')
                   AND a.inceleme_id = (SELECT MAX(inceleme_id) FROM analist_incelemesi WHERE okuma_id = f.okuma_id)"""
        # Dışarıya sunucudaki tam yol değil, yalnızca arşivdeki dosya adı verilir
        return [dict(r) | {"dosya": Path(r["dosya"]).name} for r in con.execute(sql)]

    return app


def _varsayilan():
    from .okuyucu import SeisBenchOkuyucu
    return uygulama_olustur(os.environ.get("FAZ_DB", KOK / "servis" / "faz.db"), SeisBenchOkuyucu(),
                            os.environ.get("FAZ_ARSIV", KOK / "servis" / "arsiv"))


def __getattr__(ad):  # `uvicorn servis.api:app` çağrısında model ilk erişimde yüklenir
    if ad == "app":
        global app
        app = _varsayilan()
        return app
    raise AttributeError(ad)
