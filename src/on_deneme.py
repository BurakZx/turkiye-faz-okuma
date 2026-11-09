"""
Ön deneme: Hazır (küresel / bölgesel) faz okuma modellerinin Türkiye kayıtlarındaki başarımı.

Akış
  1. Olay listesi (USGS kimlikleri) -> ISC bülteninin ISF çıktısından KOERI (ISK) ve AFAD'ın
     kendi bültenlerinde raporladığı faz okumaları (KO istasyonlarında önce ISK)
  2. KOERI (KO ağı) istasyon üst verisi -> ISC okumasındaki istasyonla eşleşen KO istasyonları
     (aynı kod + episantr uzaklığı farkı < 5 km koşulu)
  3. KOERI EIDA FDSN servisinden 3 bileşenli dalga formu (orijin -5 s / +90 s)
  4. SeisBench hazır ağırlıkları (PhaseNet, EQTransformer; original/stead/instance) ile faz okuma
  5. Analist okumasıyla eşleştirme: P için ±0,5 s, S için ±1,0 s -> TP/FP/FN, P/R/F1, MAE, RMSE

Not: Referans okumalar KOERI ve AFAD'ın ISC'ye raporladığı bülten okumalarıdır (ISC ön bülteni,
2024/11 sonrası ISC tarafından gözden geçirilmemiştir). Bu bir ÖN DENEMEDİR; örneklem küçüktür,
sonuçlar genellenemez.
"""
from __future__ import annotations

import io
import json
import re
import math
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime, Stream, read
from obspy.geodetics import gps2dist_azimuth

warnings.filterwarnings("ignore")

KOK = Path(__file__).resolve().parents[1]
VERI = KOK / "veri"
SONUC = KOK / "sonuclar"
for d in (VERI / "isc", VERI / "mseed", SONUC):
    d.mkdir(parents=True, exist_ok=True)

KOERI = "http://eida-service.koeri.boun.edu.tr/fdsnws"
ISC = "https://www.isc.ac.uk/fdsnws/event/1/query"
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"

OLAYLAR = [  # USGS olay kimlikleri (2025, Batı Türkiye)
    "us7000pufs",  # M6.2 Marmara Denizi (Silivri açıkları) 23.04.2025
    "us6000qznb",  # M6.1 Sındırgı 10.08.2025
    "us6000rjsu",  # M6.0 Sındırgı 27.10.2025
    "us6000rdf2",  # M5.2 Simav 28.09.2025
    "us6000re6a",  # M5.0 Marmara Denizi 02.10.2025
    "us6000plhn",  # M5.0 Ege (Ayvacık açıkları) 21.01.2025
    "us7000pugj",  # M4.9 Marmara Denizi 23.04.2025
    "us6000rfqg",  # M4.9 Simav 08.10.2025
]
MAKS_UZAKLIK_KM = 165
TOLERANS = {"P": 0.5, "S": 1.0}
ESIK = 0.3
MODELLER = [("PhaseNet", w) for w in ("original", "stead", "instance")] + \
           [("EQTransformer", w) for w in ("original", "stead", "instance")]


def http_get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "bitirme-projesi-prototip/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def faz_sinifi(ad: str | None) -> str | None:
    if not ad:
        return None
    ad = ad.strip()
    if ad[0] == "P" and ad in {"P", "Pg", "Pn", "Pb", "P*"}:
        return "P"
    if ad[0] == "S" and ad in {"S", "Sg", "Sn", "Sb", "S*"}:
        return "S"
    return None


def usgs_olay(evid: str) -> dict:
    j = json.loads(http_get(f"{USGS}?eventid={evid}&format=geojson"))
    lon, lat, dep = j["geometry"]["coordinates"]
    return {"evid": evid, "zaman": UTCDateTime(j["properties"]["time"] / 1000.0),
            "enlem": lat, "boylam": lon, "derinlik_km": dep, "M": j["properties"]["mag"],
            "yer": j["properties"]["place"]}


ISF_SATIR = re.compile(
    r"^(?P<sta>[A-Z0-9]{2,5})\s+(?P<dist>\d+\.\d+)\s+\d+\.\d+\s+(?P<faz>\S+)\s+"
    r"(?P<saat>\d\d:\d\d:\d\d(?:\.\d+)?)\s.*?\s(?P<arrid>\d{6,})\s+\S+\s+\S+\s+\S+\s+"
    r"(?P<ajans>[A-Z_]+)\s+(?P<yazar>[A-Z_]+)\s")
REFERANS_AJANSLAR = ("ISK", "AFAD")  # öncelik sırası: önce KOERI (ISK), yoksa AFAD


def isc_okumalar(olay: dict) -> tuple[dict, pd.DataFrame]:
    """ISC bülteninin ISF metin çıktısından, raporlayan ajansı belli faz okumaları.

    QuakeML çıktısında okumanın hangi ajanstan geldiği yazmaz; ISF çıktısında her varış
    satırında raporlayan ajans bulunur (ISK = KOERI/Kandilli, AFAD = AFAD). Referans olarak
    yalnızca bu iki Türk ajansının kendi bültenlerinde raporladığı okumalar kullanılır.
    """
    dosya = VERI / "isc" / f"{olay['evid']}.isf.html"
    t = olay["zaman"]
    if not dosya.exists():
        a, b = t - 60, t + 60
        url = ("https://www.isc.ac.uk/cgi-bin/web-db-run?request=COMPREHENSIVE&out_format=ISF2"
               f"&searchshape=CIRC&ctr_lat={olay['enlem']:.3f}&ctr_lon={olay['boylam']:.3f}"
               "&radius=60&max_dist_units=km"
               f"&start_year={a.year}&start_month={a.month}&start_day={a.day}"
               f"&start_time={a.strftime('%H:%M:%S')}"
               f"&end_year={b.year}&end_month={b.month}&end_day={b.day}"
               f"&end_time={b.strftime('%H:%M:%S')}"
               f"&min_mag={max(olay['M'] - 0.8, 3.0):.1f}&req_mag_type=Any&include_phases=on")
        dosya.write_bytes(http_get(url, timeout=240))
    satirlar = []
    for satir in dosya.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ISF_SATIR.match(satir)
        if not m or m["ajans"] not in REFERANS_AJANSLAR:
            continue
        faz = faz_sinifi(m["faz"])
        km = float(m["dist"]) * 111.195
        if faz is None or km > MAKS_UZAKLIK_KM:
            continue
        zaman = UTCDateTime(f"{t.date}T{m['saat']}")
        if zaman < t - 3600:          # gece yarısını geçen varışlar
            zaman += 86400
        satirlar.append({"istasyon": m["sta"], "faz": faz, "isf_faz": m["faz"], "zaman": zaman,
                         "uzaklik_km": km, "ajans": m["ajans"]})
    df = pd.DataFrame(satirlar)
    if df.empty:
        return {}, df
    # aynı istasyon+faz için önce ISK, yoksa AFAD; aynı ajansta en erken okuma (ilk varış)
    df["oncelik"] = df.ajans.map({a: i for i, a in enumerate(REFERANS_AJANSLAR)})
    df["ts"] = df.zaman.map(float)
    df = df.sort_values(["oncelik", "ts"]).groupby(["istasyon", "faz"], as_index=False).first()
    return {"isc_zaman": t, "isc_enlem": olay["enlem"], "isc_boylam": olay["boylam"]}, \
        df.drop(columns=["oncelik", "ts"])


def ko_istasyonlari() -> pd.DataFrame:
    dosya = VERI / "ko_istasyonlar.txt"
    if not dosya.exists():
        dosya.write_bytes(http_get(f"{KOERI}/station/1/query?network=KO&level=station&format=text"))
    df = pd.read_csv(dosya, sep="|")
    df.columns = [c.strip("# ").strip() for c in df.columns]
    return df.drop_duplicates("Station", keep="last").set_index("Station")


def dalga_formu_indir(evid: str, sta: str, t0: UTCDateTime) -> Stream | None:
    dosya = VERI / "mseed" / f"{evid}_{sta}.mseed"
    if dosya.exists():
        if dosya.stat().st_size == 0:
            return None
        st = read(str(dosya))
        return st if len(st) else None
    for kanal in ("HH?", "BH?", "EH?"):
        url = (f"{KOERI}/dataselect/1/query?network=KO&station={sta}&channel={kanal}"
               f"&starttime={(t0 - 5).isoformat()}&endtime={(t0 + 90).isoformat()}")
        try:
            veri = http_get(url, timeout=60)
        except Exception:
            veri = b""
        if not veri:
            continue
        st = read(io.BytesIO(veri))
        st.merge(method=1, fill_value=0)
        konum = sorted({tr.stats.location for tr in st})[0]
        st = Stream([tr for tr in st if tr.stats.location == konum])
        if len({tr.stats.channel[-1] for tr in st}) >= 3:
            st.write(str(dosya), format="MSEED")
            return st
    dosya.write_bytes(b"")  # bulunamadı, tekrar denenmesin
    return None


def eslestir(ref: pd.DataFrame, tahmin: pd.DataFrame) -> tuple[list, int]:
    """Her referans okuma için aynı istasyon+fazdaki en yakın, kullanılmamış model okuması."""
    kullanildi = set()
    kayitlar = []
    for r in ref.itertuples():
        aday = tahmin[(tahmin.istasyon == r.istasyon) & (tahmin.faz == r.faz)]
        en_iyi, fark = None, None
        for idx, t in aday.iterrows():
            if idx in kullanildi:
                continue
            d = float(t.zaman - r.zaman)
            if fark is None or abs(d) < abs(fark):
                en_iyi, fark = idx, d
        tp = fark is not None and abs(fark) <= TOLERANS[r.faz]
        if tp:
            kullanildi.add(en_iyi)
        kayitlar.append({"istasyon": r.istasyon, "faz": r.faz, "uzaklik_km": r.uzaklik_km,
                         "artik_s": fark, "tp": tp})
    # FP yalnızca analistin o fazı okuduğu istasyonlarda sayılır: analistin S okumadığı bir
    # istasyondaki model S okuması doğru da olabilir, yanlış da; referans olmadan yargılanamaz.
    degerlendirilebilir = set(zip(ref.istasyon, ref.faz))
    fp = {f: int(sum(1 for idx, t in tahmin.iterrows()
                     if t.faz == f and idx not in kullanildi
                     and (t.istasyon, t.faz) in degerlendirilebilir)) for f in ("P", "S")}
    return kayitlar, fp


def main():
    import seisbench.models as sbm
    import torch
    torch.set_num_threads(max(1, (torch.get_num_threads())))

    ko = ko_istasyonlari()
    olaylar, referanslar, akislar = [], [], {}
    for evid in OLAYLAR:
        o = usgs_olay(evid)
        isc, ref = isc_okumalar(o)
        if ref.empty:
            print(f"[{evid}] ISC'de yakın okuma yok, atlandı"); continue
        o.update(isc)
        st_olay = Stream()
        tutulan = []
        for sta in sorted(ref.istasyon.unique()):
            if sta not in ko.index:
                continue
            km_ko = gps2dist_azimuth(o["isc_enlem"], o["isc_boylam"],
                                     ko.loc[sta, "Latitude"], ko.loc[sta, "Longitude"])[0] / 1000
            km_isc = ref[ref.istasyon == sta].uzaklik_km.iloc[0]
            if abs(km_ko - km_isc) > 5:
                continue  # aynı kodlu farklı istasyon olabilir
            st = dalga_formu_indir(evid, sta, o["isc_zaman"])
            if st is None:
                continue
            st_olay += st
            tutulan.append(sta)
        ref = ref[ref.istasyon.isin(tutulan)].assign(evid=evid)
        print(f"[{evid}] M{o['M']} {o['yer']}: {len(tutulan)} KO istasyonu, "
              f"{(ref.faz == 'P').sum()} P + {(ref.faz == 'S').sum()} S referans okuma")
        if len(tutulan):
            olaylar.append(o); referanslar.append(ref); akislar[evid] = st_olay

    ref_tum = pd.concat(referanslar, ignore_index=True)
    pd.DataFrame([{k: str(v) if isinstance(v, UTCDateTime) else v for k, v in o.items()} for o in olaylar]) \
        .to_csv(SONUC / "olaylar.csv", index=False)
    ref_tum.assign(zaman=ref_tum.zaman.astype(str)).to_csv(SONUC / "referans_okumalar.csv", index=False)

    ozet, eslesme_tum, fp_olay = [], [], []  # fp_olay: olay düzeyinde F1 bootstrap'ı için
    for mimari, agirlik in MODELLER:
        ad = f"{mimari}-{agirlik}"
        model = getattr(sbm, mimari).from_pretrained(agirlik)
        model.eval()
        kayitlar, fp_top, sure = [], {"P": 0, "S": 0}, 0.0
        for o in olaylar:
            st = akislar[o["evid"]]
            t = time.perf_counter()
            cikti = model.classify(st.copy(), P_threshold=ESIK, S_threshold=ESIK)
            sure += time.perf_counter() - t
            pk = pd.DataFrame([{"istasyon": p.trace_id.split(".")[1], "faz": p.phase,
                                "zaman": p.peak_time} for p in cikti.picks])
            if pk.empty:
                pk = pd.DataFrame(columns=["istasyon", "faz", "zaman"])
            ref = ref_tum[ref_tum.evid == o["evid"]]
            k, fp = eslestir(ref, pk)
            for x in k:
                x.update(model=ad, evid=o["evid"])
            kayitlar += k
            for f in fp_top:
                fp_top[f] += fp[f]
                fp_olay.append({"model": ad, "evid": o["evid"], "faz": f, "FP": fp[f]})
        df = pd.DataFrame(kayitlar)
        eslesme_tum.append(df)
        for faz in ("P", "S"):
            d = df[df.faz == faz]
            tp = int(d.tp.sum()); fn = len(d) - tp
            # FP'yi faz bazında ayırmak için: modelin o fazdaki toplam okuması - TP
            hata = d[d.tp].artik_s.astype(float)
            fp = fp_top[faz]
            prec = tp / (tp + fp) if tp + fp else np.nan
            rec = tp / len(d) if len(d) else np.nan
            ozet.append({"model": ad, "faz": faz, "referans": len(d), "TP": tp, "FP": fp, "FN": fn,
                         "precision": prec, "recall": rec,
                         "F1": 2 * prec * rec / (prec + rec) if prec and rec else 0.0,
                         "MAE_s": hata.abs().mean(), "RMSE_s": math.sqrt((hata ** 2).mean()) if len(hata) else np.nan,
                         "ortalama_artik_s": hata.mean(), "sure_s": sure})
        print(f"  {ad}: FP={fp_top}, süre={sure:.1f} s")

    ozet_df = pd.DataFrame(ozet)
    ozet_df.to_csv(SONUC / "ozet_metrikler.csv", index=False)
    pd.concat(eslesme_tum).to_csv(SONUC / "eslesmeler.csv", index=False)
    pd.DataFrame(fp_olay).to_csv(SONUC / "fp_olay.csv", index=False)
    print(ozet_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
