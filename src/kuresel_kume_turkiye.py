"""Küresel eğitim kümelerinde (STEAD, GEOFON, NEIC) Türkiye kayıtlarının payı.

SeisBench'in metadata.csv dosyaları (yalnızca üst veri, dalga formu yok) indirilip okunur.
Türkiye kaydı iki biçimde sayılır: (a) kaynağı Türkiye kutusunda (35,5–42,5 K, 25,5–45 D) olan izler,
(b) istasyonu KO veya TU ağında olan izler.
Kullanım: python kuresel_kume_turkiye.py <metadata klasörü>   (klasörde stead.csv, geofon.csv, neic.csv)
Çıktı: sonuclar/kuresel_kumelerde_turkiye.csv
"""
import sys
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parents[1]
KUTU = dict(lat=(35.5, 42.5), lon=(25.5, 45.0))


def main(klasor: Path):
    satir = []
    for ad in ("stead", "geofon", "neic"):
        yol = klasor / f"{ad}.csv"
        bas = pd.read_csv(yol, nrows=0).columns
        kol = [c for c in ("source_latitude_deg", "source_longitude_deg", "station_network_code") if c in bas]
        df = pd.read_csv(yol, usecols=kol or ["trace_name"], low_memory=False)  # NEIC'de konum yok
        d = {"kume": ad.upper(), "iz_sayisi": len(df)}
        if "source_latitude_deg" in df:
            kaynak_tr = df.source_latitude_deg.between(*KUTU["lat"]) & df.source_longitude_deg.between(*KUTU["lon"])
            d["kaynak_turkiye"] = int(kaynak_tr.sum())
            d["kaynak_turkiye_yuzde"] = round(100 * kaynak_tr.mean(), 2)
        if "station_network_code" in df:
            ag_tr = df.station_network_code.isin(["KO", "TU"])
            d["istasyon_KO_TU"] = int(ag_tr.sum())
            d["istasyon_KO_TU_yuzde"] = round(100 * ag_tr.mean(), 2)
        satir.append(d)
    out = pd.DataFrame(satir)
    out.to_csv(KOK / "sonuclar" / "kuresel_kumelerde_turkiye.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
