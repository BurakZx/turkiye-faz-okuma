"""FazOkuyucu arayüzü ve SeisBench tabanlı uygulaması.

Sınıf diyagramındaki (Şekil: UML sınıf diyagramı) FazOkuyucu arayüzünün çalışan karşılığıdır:
servis katmanı yalnızca bu arayüzü bilir, böylece eğitilmiş CRNN ileride aynı yere takılabilir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from obspy import Stream, UTCDateTime


@dataclass
class Okuma:
    istasyon: str
    faz: str            # "P" | "S"
    zaman: UTCDateTime
    olasilik: float


class FazOkuyucu(Protocol):
    mimari: str
    agirlik: str

    def oku(self, akis: Stream) -> list[Okuma]: ...


class SeisBenchOkuyucu:
    """SeisBench'teki hazır bir modelle (varsayılan: PhaseNet, STEAD ağırlıkları) faz okur."""

    def __init__(self, mimari: str = "PhaseNet", agirlik: str = "stead", esik: float = 0.3):
        import seisbench.models as sbm  # ağır içe aktarım: yalnızca gerçek okuyucu kullanılırsa

        self.mimari, self.agirlik, self.esik = mimari, agirlik, esik
        self._model = getattr(sbm, mimari).from_pretrained(agirlik)
        self._model.eval()

    def oku(self, akis: Stream) -> list[Okuma]:
        cikti = self._model.classify(akis.copy(), P_threshold=self.esik, S_threshold=self.esik)
        return [Okuma(p.trace_id.split(".")[1], p.phase, p.peak_time, float(p.peak_value)) for p in cikti.picks]
