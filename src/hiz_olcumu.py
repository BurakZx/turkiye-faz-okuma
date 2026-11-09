"""
Modellerin CPU üzerinde çıkarım süresi ve parametre sayısı (işlevsel olmayan gereksinim: gecikme).
Ağırlıkların eğitilmiş olması süreyi etkilemez; CRNN rastgele ağırlıklarla ölçülür.
"""
import platform
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from crnn_model import FazCRNN  # noqa: E402

import seisbench.models as sbm  # noqa: E402

torch.manual_seed(0)
SONUC = Path(__file__).resolve().parents[1] / "sonuclar"


def olc(model, n_ornek, batch, tekrar=20):
    x = torch.randn(batch, 3, n_ornek)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        sureler = []
        for _ in range(tekrar):
            t = time.perf_counter()
            model(x)
            sureler.append((time.perf_counter() - t) * 1000)
    return statistics.median(sureler)


modeller = [
    ("Özgün CRNN", FazCRNN(), 3001),
    ("PhaseNet", sbm.PhaseNet(), 3001),
    ("EQTransformer", sbm.EQTransformer(), 6000),
]
satir = []
for ad, m, n in modeller:
    m.eval()
    satir.append({
        "model": ad,
        "pencere_s": n / 100,
        "parametre": sum(p.numel() for p in m.parameters()),
        "ms_batch1": round(olc(m, n, 1), 1),
        "ms_batch32": round(olc(m, n, 32), 1),
    })
df = pd.DataFrame(satir)
df["ms_pencere_basina_b32"] = (df.ms_batch32 / 32).round(2)
df.to_csv(SONUC / "hiz_olcumu.csv", index=False)
print(f"CPU: {platform.processor() or platform.machine()}, iş parçacığı: {torch.get_num_threads()}")
print(df.to_string(index=False))
