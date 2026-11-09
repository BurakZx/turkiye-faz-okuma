"""Ön denemedeki S gecikmesinin olay, bölge ve faz türüne göre ayrıştırılması.

Dış değerlendirmede iki soru soruldu: (1) S okumalarının yarısı tek bir depremden geliyor;
gecikme bu olayın davranışı olabilir mi? (2) Pg/Pn ve Sg/Sn birleştirildiği için uzak
istasyonlarda referans daha erken bir faza (Sn) kayıyor olabilir mi?

Bu betik, yalnızca tolerans içindeki (TP) okumaların artıklarını kullanarak:
  - olay bazında ortalama artık tablosu,
  - olaylar yeniden örneklenerek (olay düzeyinde bootstrap) %95 güven aralıkları,
  - Sg/Sn ve Marmara/diğer ayrımı
üretir.  Çıktılar: sonuclar/olay_bazli_artik.csv, sonuclar/artik_ayristirma.csv,
sekiller/olay_bazli_s_artik.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
SONUC, SEKIL = KOK / "sonuclar", KOK / "sekiller"
MARMARA = {"us7000pufs", "us6000re6a", "us7000pugj"}  # Marmara Denizi depremleri
KISA_YER = {"us7000pufs": "Marmara Denizi", "us6000re6a": "Marmara Denizi", "us7000pugj": "Marmara Denizi",
            "us6000qznb": "Sındırgı", "us6000rjsu": "Sındırgı", "us6000rdf2": "Simav",
            "us6000rfqg": "Emet", "us6000plhn": "Ayvacık açıkları"}
B_SAYISI, TOHUM = 5000, 0


def olay_bootstrap(g: pd.DataFrame, rng) -> tuple[float, float]:
    """Olayları yerine koyarak yeniden örnekle; ortalama artığın %95 güven aralığı."""
    gruplar = {e: d.artik_s.to_numpy() for e, d in g.groupby("evid")}
    olaylar = list(gruplar)
    ort = []
    for _ in range(B_SAYISI):
        secim = rng.choice(len(olaylar), len(olaylar))
        ort.append(np.concatenate([gruplar[olaylar[i]] for i in secim]).mean())
    return float(np.percentile(ort, 2.5)), float(np.percentile(ort, 97.5))


def main():
    esl = pd.read_csv(SONUC / "eslesmeler.csv")
    ref = pd.read_csv(SONUC / "referans_okumalar.csv")
    olay = pd.read_csv(SONUC / "olaylar.csv")
    tp = esl[esl.tp].merge(ref[["istasyon", "faz", "evid", "isf_faz"]], on=["istasyon", "faz", "evid"], how="left")
    tp["marmara"] = tp.evid.isin(MARMARA)
    rng = np.random.default_rng(TOHUM)

    # 1) olay bazında S artığı (model ortalamaları) + referans S sayısı
    s = tp[tp.faz == "S"]
    tab = s.groupby(["evid", "model"]).artik_s.mean().unstack()
    tab["model_medyani"] = tab.median(axis=1)
    tab["S_referans"] = ref[ref.faz == "S"].groupby("evid").size()
    tab = tab.join(olay.set_index("evid")[["zaman", "M"]])
    tab["yer"] = [KISA_YER[e] for e in tab.index]
    tab["bolge"] = np.where(tab.index.isin(MARMARA), "Marmara", "Batı Anadolu / Ege")
    tab.sort_values(["bolge", "zaman"]).to_csv(SONUC / "olay_bazli_artik.csv")

    # 2) model × faz ayrıştırması
    satir = []
    for (m, faz), g in tp.groupby(["model", "faz"]):
        alt, ust = olay_bootstrap(g, rng)
        d = {"model": m, "faz": faz, "ortalama": g.artik_s.mean(), "medyan": g.artik_s.median(),
             "olay_GA_alt": alt, "olay_GA_ust": ust, "n": len(g),
             "marmara_ort": g[g.marmara].artik_s.mean(), "diger_ort": g[~g.marmara].artik_s.mean(),
             "n_marmara": int(g.marmara.sum()), "n_diger": int((~g.marmara).sum())}
        for f in ("g", "n"):
            alt_faz = g[g.isf_faz.str.lower().str.endswith(f)]
            d[f"{faz}{f}_ort"], d[f"{faz}{f}_n"] = alt_faz.artik_s.mean(), len(alt_faz)
        satir.append(d)
    ay = pd.DataFrame(satir)
    ay.round(3).to_csv(SONUC / "artik_ayristirma.csv", index=False)
    print(ay[ay.faz == "S"][["model", "ortalama", "olay_GA_alt", "olay_GA_ust", "marmara_ort", "diger_ort",
                             "Sg_ort", "Sn_ort"]].round(3).to_string(index=False))

    # 2b) F1 için olay düzeyinde bootstrap (TP/FN eslesmeler.csv'den, FP fp_olay.csv'den)
    fp_olay = pd.read_csv(SONUC / "fp_olay.csv")
    say = (esl.assign(TP=esl.tp.astype(int), FN=(~esl.tp).astype(int))
              .groupby(["model", "faz", "evid"])[["TP", "FN"]].sum().reset_index()
              .merge(fp_olay, on=["model", "faz", "evid"], how="left").fillna({"FP": 0}))
    f1_satir = []
    for (m, faz), g in say.groupby(["model", "faz"]):
        g = g.set_index("evid")
        f1 = lambda d: 2 * d.TP.sum() / (2 * d.TP.sum() + d.FP.sum() + d.FN.sum())
        bs = [f1(g.iloc[rng.choice(len(g), len(g))]) for _ in range(B_SAYISI)]
        f1_satir.append({"model": m, "faz": faz, "F1": f1(g), "GA_alt": np.percentile(bs, 2.5),
                         "GA_ust": np.percentile(bs, 97.5)})
    f1_df = pd.DataFrame(f1_satir)
    f1_df.round(3).to_csv(SONUC / "f1_guven_araligi.csv", index=False)
    print(f1_df.round(2).to_string(index=False))

    # 3) şekil: olay bazında S artığı
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 200, "savefig.bbox": "tight"})
    t = tab.sort_values(["bolge", "model_medyani"], ascending=[False, True])
    modeller = [c for c in tab.columns if c.startswith(("PhaseNet", "EQTransformer"))]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.axvline(0, color="#52514e", lw=0.8)
    for i, (e, r) in enumerate(t.iterrows()):
        renk = "#c0392b" if e in MARMARA else "#52514e"
        if e in MARMARA:
            ax.axhspan(i - 0.45, i + 0.45, color="#fbe9e7", zorder=0)
        ax.scatter(r[modeller].astype(float), [i] * len(modeller), s=14, color="#9a9893", zorder=2,
                   label="tek model" if i == 0 else None)
        ax.scatter([r.model_medyani], [i], marker="D", s=46, color=renk, zorder=3,
                   label=("Marmara (model medyanı)" if e == "us7000pufs" else
                          "diğer (model medyanı)" if e == "us6000rfqg" else None))
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f"{r.yer}  M{str(r.M).replace('.', ',')}  (S={int(r.S_referans)})" for _, r in t.iterrows()],
                       fontsize=9)
    ax.set_xlim(-0.6, 1.05)
    ax.set_xlabel("Ortalama S artığı (s): model − KOERI okuması (yalnızca ±1 s tolerans içindeki okumalar)")
    ax.grid(axis="x", color="#e4e3df", lw=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(SEKIL / "olay_bazli_s_artik.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
