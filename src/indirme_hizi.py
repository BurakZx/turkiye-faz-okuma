"""KOERI FDSN servisinden toplu (bulk POST) ve tek tek dalga formu indirme hızını ölçer.

Asıl veri kümesinin (~1,4 milyon olay-istasyon kaydı) kaç günde indirilebileceğini tahmin etmek için
küçük ve nazik bir deneme yapılır: tek bir olay için 20 istasyon, 95 s'lik 3 bileşenli pencereler.
Çıktı: sonuclar/indirme_hizi.csv
"""
import time
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

KOK = Path(__file__).resolve().parents[1]
ORIJIN = UTCDateTime("2025-10-27T19:48:28")  # Sındırgı M6,0 (ön deneme olayı)
N_ISTASYON = 20


def main():
    istemci = Client("http://eida-service.koeri.boun.edu.tr", timeout=120)
    ist = [s.code for s in istemci.get_stations(network="KO", level="station",
                                                 latitude=39.19, longitude=28.24, maxradius=1.8)[0]]
    ist = sorted(ist)[:N_ISTASYON]
    t0, t1 = ORIJIN - 5, ORIJIN + 90
    satir = []

    bas = time.perf_counter()
    st = istemci.get_waveforms_bulk([("KO", s, "*", "HH?", t0, t1) for s in ist])
    sure = time.perf_counter() - bas
    n = len({tr.stats.station for tr in st})
    mb = sum(tr.data.nbytes for tr in st) / 1e6
    satir.append({"yontem": "bulk POST (tek istek)", "kayit": n, "sure_s": round(sure, 2), "MB_float": round(mb, 2)})

    bas, n2 = time.perf_counter(), 0
    for s in ist[:5]:
        try:
            n2 += len(istemci.get_waveforms("KO", s, "*", "HH?", t0, t1)) > 0
        except Exception:
            pass
    sure2 = time.perf_counter() - bas
    satir.append({"yontem": "tek tek GET (5 istasyon)", "kayit": n2, "sure_s": round(sure2, 2), "MB_float": None})

    df = pd.DataFrame(satir)
    df["kayit_per_s"] = (df.kayit / df.sure_s).round(2)
    hedef = 1_367_325  # veri_hacmi_tahmini.csv: P_200 (≈ olay-istasyon kaydı)
    df["tahmini_gun_1_akis"] = (hedef / df.kayit_per_s / 86400).round(1)
    df.to_csv(KOK / "sonuclar" / "indirme_hizi.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
