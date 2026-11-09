"""Asıl veri kümesinin büyüklüğünü tahmin eder ve H3 için kaba bir güç analizi yapar.

2020–2025 arasından sabit tohumla seçilen rastgele haftalar için ISC bülteninin ISF2 çıktısı
(Türkiye ve yakın çevresi, 35,5–42,5 K / 25,5–45 D) indirilir. Her haftada ISK (KOERI)
tarafından raporlanan P ve S okumaları, uzaklık bandına ve bölgeye göre sayılır. Bölge: episantrı
Doğu Anadolu Fay Zonu'nun ya da 2023 Çardak–Sürgü fayının yaklaşık izine 75 km'den yakın olan
olaylar "DAFZ", diğerleri "diğer" (fay izleri kaba nokta dizileridir; yalnızca hacim tahmini içindir), haftalık ortalama 6 yıla ölçeklenir. Belirsizlik, haftalar üzerinden bootstrap ile verilir.

Çıktılar: sonuclar/veri_hacmi_tahmini.csv, sonuclar/guc_analizi.csv
Çalıştırma: .venv/bin/python src/veri_hacmi_tahmini.py   (~10 dk; indirmeler veri/isc_hacim/ altında önbelleklenir)
"""
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from on_deneme import ISF_SATIR, faz_sinifi, http_get

KOK = Path(__file__).resolve().parents[1]
ONBELLEK = KOK / "veri" / "isc_hacim"
SONUC = KOK / "sonuclar"
HAFTA_SAYISI, TOHUM = 12, 2026
BOLGE = dict(bot_lat=35.5, top_lat=42.5, left_lon=25.5, right_lon=45.0)
# Yaklaşık fay izleri (enlem, boylam): DAFZ Karlıova → Antakya ve 2023 Çardak–Sürgü fayı
FAY_IZLERI = [
    [(39.30, 41.00), (38.70, 39.95), (38.45, 39.30), (38.20, 38.87), (38.03, 38.25), (37.78, 37.65),
     (37.38, 36.85), (36.50, 36.36), (36.20, 36.16)],
    [(38.02, 36.48), (38.06, 37.19), (38.10, 38.00)],
]
DAFZ_KM = 75
ORIJIN = re.compile(r"^\d{4}/\d\d/\d\d\s+\d\d:\d\d:\d\d\.\d+\s+.*?\s(-?\d+\.\d+)\s+(-?\d+\.\d+)\s")


def haftalar() -> list[dt.date]:
    rng = np.random.default_rng(TOHUM)
    bas, gun = dt.date(2020, 1, 1), (dt.date(2025, 12, 24) - dt.date(2020, 1, 1)).days
    return sorted(bas + dt.timedelta(days=int(g)) for g in rng.choice(gun, HAFTA_SAYISI, replace=False))


def hafta_isf(bas: dt.date) -> str:
    dosya = ONBELLEK / f"{bas.isoformat()}.isf.html"
    if not dosya.exists() or dosya.stat().st_size == 0:
        son = bas + dt.timedelta(days=6)
        url = ("https://www.isc.ac.uk/cgi-bin/web-db-run?request=COMPREHENSIVE&out_format=ISF2&searchshape=RECT"
               + "".join(f"&{k}={v}" for k, v in BOLGE.items())
               + f"&start_year={bas.year}&start_month={bas.month}&start_day={bas.day}&start_time=00:00:00"
               + f"&end_year={son.year}&end_month={son.month}&end_day={son.day}&end_time=23:59:59"
               + "&include_phases=on&include_magnitudes=on&include_links=off&include_headers=on")
        dosya.write_bytes(http_get(url, timeout=600))
    return dosya.read_text(encoding="utf-8", errors="replace")


def fay_uzakligi_km(enlem: float, boylam: float) -> float:
    """Noktanın fay izlerine en kısa uzaklığı (yerel eşdikdörtgen izdüşüm; ~1 km doğruluk yeterli)."""
    kx = 111.195 * np.cos(np.radians(enlem))
    p = np.array([boylam * kx, enlem * 111.195])
    en_kisa = np.inf
    for iz in FAY_IZLERI:
        q = np.array([(lo * kx, la * 111.195) for la, lo in iz])
        for a, b in zip(q[:-1], q[1:]):
            t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
            en_kisa = min(en_kisa, np.linalg.norm(p - (a + t * (b - a))))
    return float(en_kisa)


def say(metin: str) -> dict:
    """Bir haftalık ISF metninde ISK okumalarını say (aynı olay+istasyon+faz bir kez)."""
    goruldu, olay, konum, sayac = set(), None, None, {}
    for satir in metin.splitlines():
        if satir.startswith("Event "):
            olay, konum = satir.split()[1], None
            continue
        if konum is None and (m := ORIJIN.match(satir)):
            konum = (float(m.group(1)), float(m.group(2)))
            continue
        m = ISF_SATIR.match(satir)
        if not m or m["ajans"] != "ISK" or olay is None:
            continue
        faz = faz_sinifi(m["faz"])
        km = float(m["dist"]) * 111.195
        if faz is None or km > 200 or (olay, m["sta"], faz) in goruldu:
            continue
        goruldu.add((olay, m["sta"], faz))
        bolge = "dafz" if konum and fay_uzakligi_km(*konum) <= DAFZ_KM else "diger"
        for anahtar in (f"{faz}_200", f"{faz}_{bolge}_200") + ((f"{faz}_165",) if km <= 165 else ()):
            sayac[anahtar] = sayac.get(anahtar, 0) + 1
    sayac["olay_isk"] = len({o for o, _, _ in goruldu})
    return sayac


def guc(delta: float, psi: float, alfa=0.05, guc_=0.80) -> int:
    """Eşleştirilmiş oranlar için McNemar testi örneklem büyüklüğü (Connor, 1987)."""
    za, zb = norm.ppf(1 - alfa / 2), norm.ppf(guc_)
    return int(np.ceil((za * np.sqrt(psi) + zb * np.sqrt(psi - delta ** 2)) ** 2 / delta ** 2))


def main():
    ONBELLEK.mkdir(parents=True, exist_ok=True)
    satirlar = []
    for h in haftalar():
        s = say(hafta_isf(h))
        s["hafta"] = h.isoformat()
        satirlar.append(s)
        print(h, {k: v for k, v in s.items() if k != "hafta"}, flush=True)
    df = pd.DataFrame(satirlar).fillna(0).set_index("hafta")
    df.to_csv(SONUC / "veri_hacmi_haftalik.csv")

    olcek = 52.18 * 6  # 2020–2025, 6 yıl
    rng = np.random.default_rng(0)
    ozet = []
    for k in ["P_165", "S_165", "P_200", "S_200", "P_diger_200", "P_dafz_200", "S_diger_200", "S_dafz_200", "olay_isk"]:
        x = df[k].to_numpy() if k in df else np.zeros(len(df))
        bs = [rng.choice(x, len(x)).mean() * olcek for _ in range(5000)]
        ozet.append({"olcut": k, "haftalik_ortalama": round(x.mean(), 1), "haftalik_medyan": float(np.median(x)),
                     "tahmin_6_yil": int(round(x.mean() * olcek)), "medyanla_6_yil": int(round(np.median(x) * olcek)),
                     "gA_alt": int(np.percentile(bs, 2.5)), "gA_ust": int(np.percentile(bs, 97.5))})
    oz = pd.DataFrame(ozet)
    oz.to_csv(SONUC / "veri_hacmi_tahmini.csv", index=False)
    print(oz.to_string(index=False))

    # H3 güç analizi: P duyarlılığında 0,05'lik farkı eşleştirilmiş okumalarda ayırt etmek için gereken
    # test okuması sayısı; olay içi bağımlılık için tasarım etkisi (DE) 2 varsayılmıştır.
    g = pd.DataFrame([{"fark": d, "uyumsuz_oran": psi, "gerekli_okuma": guc(d, psi),
                       "DE2_ile": 2 * guc(d, psi)} for d in (0.05, 0.02) for psi in (0.10, 0.15, 0.20)])
    g.to_csv(SONUC / "guc_analizi.csv", index=False)
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
