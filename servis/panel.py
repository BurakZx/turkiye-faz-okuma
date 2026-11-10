"""Dikey dilim: analist paneli (Streamlit). API'ye HTTP üzerinden bağlanır.

Çalıştırma:  .venv/bin/streamlit run servis/panel.py   (önce API: uvicorn servis.api:app --port 8000)
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from obspy import UTCDateTime, read

API = os.environ.get("FAZ_API", "http://127.0.0.1:8000")
REF = Path(__file__).resolve().parents[1] / "sonuclar" / "referans_okumalar.csv"
st.set_page_config(page_title="Faz okuma - analist paneli", layout="wide")


@st.cache_data
def referanslar() -> pd.DataFrame:
    return pd.read_csv(REF) if REF.exists() else pd.DataFrame(columns=["istasyon", "faz", "zaman"])


def virgul(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def kuyruk() -> list[dict]:
    return httpx.get(f"{API}/okumalar", params={"durum": "bekliyor"}, timeout=10).json()


st.title("Faz okuma - analist paneli")
st.caption(f"Dikey dilim prototipi · API: {API} · Eşik altındaki (olasılık < 0,7) okumalar inceleme kuyruğundadır.")
try:
    tum = httpx.get(f"{API}/okumalar", timeout=10).json()
except httpx.HTTPError:
    st.error("API'ye ulaşılamadı. Önce `uvicorn servis.api:app --port 8000` çalıştırın.")
    st.stop()

durumlar = pd.Series([o["durum"] for o in tum]).value_counts()
c = st.columns(4)
for kol, (ad, anahtar) in zip(c, [("Bekleyen", "bekliyor"), ("Otomatik kabul", "otomatik_kabul"),
                                  ("Onaylanan / düzeltilen", None), ("Reddedilen", "reddedildi")]):
    deger = int(durumlar.get("onaylandi", 0) + durumlar.get("duzeltildi", 0)) if anahtar is None else int(durumlar.get(anahtar, 0))
    kol.metric(ad, deger)

bekleyen = kuyruk()
sol, orta = st.columns([1, 3])
with sol:
    st.subheader("İnceleme kuyruğu")
    if not bekleyen:
        st.success("Kuyruk boş.")
        st.stop()
    etiket = {o["okuma_id"]: f"{o['ag']}.{o['istasyon']} · {o['faz']} · p={virgul(o['olasilik'])}" for o in bekleyen}
    secili = st.radio("Okuma", list(etiket), format_func=etiket.get, label_visibility="collapsed")
o = next(x for x in bekleyen if x["okuma_id"] == secili)
t = UTCDateTime(o["zaman"])

with orta:
    st.subheader(f"{o['ag']}.{o['istasyon']} - {o['faz']} okuması, olasılık {virgul(o['olasilik'])}")
    akis = read(o["dosya"]).slice(t - 8, t + 12)
    akis.detrend("demean").filter("bandpass", freqmin=1, freqmax=20)
    fig, axes = plt.subplots(len(akis), 1, figsize=(10, 3.6), sharex=True)
    r = referanslar()
    ref = r[(r.istasyon == o["istasyon"]) & (r.faz == o["faz"])]
    ref = [UTCDateTime(z) for z in ref.zaman if abs(UTCDateTime(z) - t) < 10]
    for ax, tr in zip(axes, akis):
        x = tr.times(reftime=t)
        ax.plot(x, tr.data / (abs(tr.data).max() or 1), color="#52514e", lw=0.6)
        ax.axvline(0, color="#2a78d6", lw=1.6, label="model okuması")
        for z in ref:
            ax.axvline(z - t, color="#eb6834", lw=1.4, ls="--", label="KOERI bülten okuması")
        ax.set_ylabel(tr.stats.channel, rotation=0, labelpad=18)
        ax.set_yticks([])
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("Model okumasına göre süre (s)")
    st.pyplot(fig)

    analist = st.text_input("Analist", value="analist1")
    k1, k2, k3 = st.columns([1, 1, 2])
    if k1.button("Onayla", type="primary", use_container_width=True):
        httpx.post(f"{API}/okumalar/{secili}/inceleme", json={"analist": analist, "karar": "onayla"})
        st.rerun()
    if k2.button("Reddet", use_container_width=True):
        httpx.post(f"{API}/okumalar/{secili}/inceleme", json={"analist": analist, "karar": "reddet"})
        st.rerun()
    with k3:
        kayma = st.number_input("Düzeltme (s, + geç)", value=round(ref[0] - t, 2) if ref else 0.0, step=0.05)
        if st.button("Düzelt ve kaydet", use_container_width=True):
            httpx.post(f"{API}/okumalar/{secili}/inceleme",
                       json={"analist": analist, "karar": "duzelt", "duzeltilmis_zaman": str(t + kayma)})
            st.rerun()
