"""Ön deneme sonuçlarından rapor şekillerini üretir (sonuclar/*.csv -> sekiller/*.png)."""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

warnings.filterwarnings("ignore")
KOK = Path(__file__).resolve().parents[1]
SONUC, SEKIL, VERI = KOK / "sonuclar", KOK / "sekiller", KOK / "veri"
SEKIL.mkdir(exist_ok=True)

P_RENK, S_RENK = "#2a78d6", "#eb6834"   # referans paletin 1. ve 2. sırası
MUREKKEP, IKINCIL, IZGARA = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": IKINCIL,
    "axes.labelcolor": MUREKKEP, "xtick.color": IKINCIL, "ytick.color": IKINCIL,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": IZGARA, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False,
})

ozet = pd.read_csv(SONUC / "ozet_metrikler.csv")
esl = pd.read_csv(SONUC / "eslesmeler.csv")
modeller = list(dict.fromkeys(ozet.model))
etiket = [m.replace("EQTransformer", "EQT").replace("-", "\n") for m in modeller]


# 1) F1 karşılaştırması (P ve S)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
x = np.arange(len(modeller)); w = 0.38
for i, (faz, renk) in enumerate((("P", P_RENK), ("S", S_RENK))):
    d = ozet[ozet.faz == faz].set_index("model").loc[modeller]
    bars = ax.bar(x + (i - 0.5) * w, d.F1, w - 0.03, color=renk, label=f"{faz} fazı", zorder=2)
    for b, v in zip(bars, d.F1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                fontsize=7.5, color=IKINCIL)
ax.set_xticks(x, etiket, fontsize=8.5)
ax.set_ylim(0, 1.08); ax.set_ylabel("F1 skoru")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", ncol=2, fontsize=9)
ax.set_title("Hazır modellerin KOERI bülten okumalarına göre F1 skoru\n"
             "(P: ±0,5 s, S: ±1,0 s tolerans)", fontsize=10, color=MUREKKEP, loc="left")
fig.savefig(SEKIL / "on_deneme_f1.png"); plt.close(fig)


# 2) Zaman artıkları (model − analist), ±3 s içinde eşleşen okumalar
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
rng = np.random.default_rng(0)
for ax, (faz, renk) in zip(axes, (("P", P_RENK), ("S", S_RENK))):
    tol = 0.5 if faz == "P" else 1.0
    ax.axhspan(-tol, tol, color=renk, alpha=0.08, zorder=0)
    ax.axhline(0, color=IKINCIL, lw=0.8)
    for i, m in enumerate(modeller):
        a = esl[(esl.model == m) & (esl.faz == faz)].artik_s.dropna()
        a = a[a.abs() <= 3]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(a)), a, s=9, color=renk, alpha=0.55,
                   edgecolor="white", linewidth=0.3, zorder=2)
        if len(a):
            ax.plot([i - 0.28, i + 0.28], [a.median()] * 2, color=MUREKKEP, lw=1.6, zorder=3)
    ax.set_xticks(range(len(modeller)), etiket, fontsize=7)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{faz} fazı (gölge: ±{str(tol).replace('.', ',')} s tolerans)", fontsize=9.5, loc="left", color=MUREKKEP)
axes[0].set_ylabel("Model − KOERI okuması (s)")
fig.suptitle("Faz zamanı artıkları (siyah çizgi: medyan)", x=0.06, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig(SEKIL / "on_deneme_artiklar.png"); plt.close(fig)


# 3) Örnek kayıt: dalga formu + iki modelin olasılık eğrileri + analist okuması
def ornek_kayit():
    import seisbench.models as sbm
    olaylar = pd.read_csv(SONUC / "olaylar.csv")
    ref = pd.read_csv(SONUC / "referans_okumalar.csv")
    ev = olaylar.iloc[0]
    r = ref[ref.evid == ev.evid]
    ikili = r.groupby("istasyon").faz.nunique()
    adaylar = r[r.istasyon.isin(ikili[ikili == 2].index)].sort_values("uzaklik_km")
    sta = adaylar.istasyon.iloc[min(2, len(adaylar) - 1)]
    km = adaylar[adaylar.istasyon == sta].uzaklik_km.iloc[0]
    st = read(str(VERI / "mseed" / f"{ev.evid}_{sta}.mseed"))
    st.detrend("demean").filter("bandpass", freqmin=1, freqmax=20)
    t0 = UTCDateTime(ev.isc_zaman)
    refP = UTCDateTime(r[(r.istasyon == sta) & (r.faz == "P")].zaman.iloc[0]) - t0
    refS = UTCDateTime(r[(r.istasyon == sta) & (r.faz == "S")].zaman.iloc[0]) - t0
    xmin, xmax = max(-2, refP - 8), refS + 15

    fig, axes = plt.subplots(5, 1, figsize=(7.4, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.15, 1.15]})
    for ax, bil in zip(axes[:3], "ZNE"):
        tr = st.select(component=bil)[0]
        t = tr.times(reftime=t0)
        ax.plot(t, tr.data / np.abs(tr.data).max(), color=IKINCIL, lw=0.5)
        ax.set_ylabel(bil, rotation=0, labelpad=10); ax.set_yticks([])
    for (ax, (mim, w)) in zip(axes[3:], (("PhaseNet", "instance"), ("EQTransformer", "original"))):
        model = getattr(sbm, mim).from_pretrained(w)
        ann = model.annotate(st.copy())
        for kanal, renk, ad in (("P", P_RENK, "P olasılığı"), ("S", S_RENK, "S olasılığı")):
            tr = [x for x in ann if x.stats.channel.endswith(f"_{kanal}")][0]
            ax.plot(tr.times(reftime=t0), tr.data, color=renk, lw=1.4, label=ad)
        ax.set_ylim(0, 1.05); ax.set_ylabel(f"{mim.replace('EQTransformer','EQT')}\n({w})", fontsize=8.5)
        ax.axhline(0.3, color=IKINCIL, lw=0.6, ls=":")
    for ax in axes:
        ax.axvline(refP, color=P_RENK, lw=1.0, ls="--")
        ax.axvline(refS, color=S_RENK, lw=1.0, ls="--")
        ax.grid(axis="y", visible=False)
    axes[3].legend(loc="upper right", fontsize=8, ncol=2)
    axes[0].text(refP, 1.08, " KOERI okuması P", color=P_RENK, fontsize=8, transform=axes[0].get_xaxis_transform())
    axes[0].text(refS, 1.08, " KOERI okuması S", color=S_RENK, fontsize=8, transform=axes[0].get_xaxis_transform())
    axes[-1].set_xlim(xmin, xmax)
    axes[-1].set_xlabel("Orijin zamanından itibaren süre (s)")
    yer = {"us7000pufs": "Marmara Denizi (Silivri açıkları)"}.get(ev.evid, ev.yer)
    fig.suptitle(f"KO.{sta} istasyonu · {yer} · M{str(ev.M).replace('.', ',')} · {t0.date} · Δ ≈ {km:.0f} km\n"
                 "(1–20 Hz filtreli; kesikli çizgi: KOERI bülten okuması; noktalı çizgi: 0,3 eşik)",
                 x=0.06, ha="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(SEKIL / "on_deneme_ornek_kayit.png"); plt.close(fig)
    return sta


print("örnek istasyon:", ornek_kayit())
print("Şekiller:", sorted(p.name for p in SEKIL.glob("*.png")))
