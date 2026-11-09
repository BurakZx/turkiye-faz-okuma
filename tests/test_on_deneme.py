"""Ön deneme hattının eşleştirme, faz sınıflandırma ve ISF ayrıştırma mantığı için birim testleri."""
import pandas as pd
from obspy import UTCDateTime

from on_deneme import ISF_SATIR, eslestir, faz_sinifi

T0 = UTCDateTime("2025-08-10T16:54:00")


def _ref(*satirlar):
    return pd.DataFrame([{"istasyon": i, "faz": f, "zaman": T0 + dt, "uzaklik_km": 50.0} for i, f, dt in satirlar])


def _tahmin(*satirlar):
    return pd.DataFrame([{"istasyon": i, "faz": f, "zaman": T0 + dt} for i, f, dt in satirlar])


def test_faz_sinifi():
    assert faz_sinifi("Pg") == "P" and faz_sinifi("Pn") == "P" and faz_sinifi("P") == "P"
    assert faz_sinifi("Sg") == "S" and faz_sinifi("Sn") == "S"
    assert faz_sinifi("PKP") is None and faz_sinifi("SPEC") is None and faz_sinifi(None) is None


def test_tolerans_icindeki_okuma_tp():
    k, fp = eslestir(_ref(("ISK", "P", 0.0)), _tahmin(("ISK", "P", 0.3)))
    assert k[0]["tp"] and abs(k[0]["artik_s"] - 0.3) < 1e-6 and fp == {"P": 0, "S": 0}


def test_tolerans_disindaki_okuma_fn_ve_fp():
    k, fp = eslestir(_ref(("ISK", "P", 0.0)), _tahmin(("ISK", "P", 0.7)))  # P toleransı 0,5 s
    assert not k[0]["tp"] and fp["P"] == 1


def test_s_toleransi_daha_genis():
    k, _ = eslestir(_ref(("ISK", "S", 0.0)), _tahmin(("ISK", "S", 0.9)))  # S toleransı 1,0 s
    assert k[0]["tp"]


def test_referanssiz_istasyon_ve_fazda_fp_sayilmaz():
    # KOERI bu istasyonda S okumamış; modelin S okuması yargılanamaz
    k, fp = eslestir(_ref(("ISK", "P", 0.0)), _tahmin(("ISK", "P", 0.1), ("ISK", "S", 5.0), ("EDC", "P", 1.0)))
    assert k[0]["tp"] and fp == {"P": 0, "S": 0}


def test_bir_model_okumasi_tek_referansla_eslesir():
    k, fp = eslestir(_ref(("ISK", "P", 0.0), ("ISK", "P", 0.2)), _tahmin(("ISK", "P", 0.1)))
    assert sum(x["tp"] for x in k) == 1 and fp["P"] == 0


def test_isf_satiri_ajans_ayristirma():
    # ISC ISF2 çıktısından gerçek bir satır (10.08.2025 Sındırgı depremi, KO.BALB)
    satir = ("BALB    0.29 338.5 Pg       16:53:55.2                                   ___          3008.0"
             "        __              974803033 ISC   IR       -- ISK   ISK   ??? ??? _  39.6400   27.8800"
             "   120.0    0.0")
    m = ISF_SATIR.match(satir)
    assert m and m["sta"] == "BALB" and m["faz"] == "Pg" and m["saat"] == "16:53:55.2" and m["ajans"] == "ISK"
