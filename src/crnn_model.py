"""
Önerilen CRNN faz okuma modeli (PyTorch).

Mimari (diyagramlar/09_crnn_mimarisi.dot ile birebir):
  Girdi 3 x 3001 (30 s, 100 Hz, Z-N-E)
  -> Çok ölçekli paralel evrişim (k = 7, 15, 31, 63; her biri 8 filtre) -> 32 kanal
  -> 3 kodlayıcı blok (adım 2): 48, 64, 96 kanal (1501, 751, 376 örnek)
  -> 2 katmanlı çift yönlü GRU (gizli 64) -> 128 öznitelik
  -> 3 çözücü blok (yukarı örnekleme + atlama bağlantısı)
  -> 1x1 evrişim + softmax -> [Gürültü, P, S] olasılık serisi (3 x 3001)

Bileşik kayıp:
  L = w_P*BCE(P) + w_S*BCE(S) + BCE(Gürültü) + lambda*(|t̂_P - t_P| + |t̂_S - t_S|)
  t̂: olasılık eğrisinin YEREL soft-argmax'ı (türevlenebilir faz zamanı, saniye). Ağırlıklı
      ortalama yalnızca en yüksek tepenin çevresindeki ±1 s'lik pencerede alınır; böylece
      (i) pencere boyunca biriken küçük taban olasılık t̂'yi pencere ortasına çekmez,
      (ii) pencerede iki olay varsa t̂ iki tepenin arasına düşmez. Pencerede faz yoksa
      (t = NaN) zaman terimi o örnek için maskelenir.

Bu dosya modelin TASARIM prototipidir; henüz eğitilmemiştir.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ORNEKLEME_HIZI = 100.0  # Hz
PENCERE = 3001          # örnek (30 s)


class CokOlcekliBlok(nn.Module):
    """Farklı frekans bantlarındaki örüntüleri ayrıştırmak için paralel evrişimler."""

    def __init__(self, giris_kanal: int = 3, filtre: int = 8, cekirdekler=(7, 15, 31, 63)):
        super().__init__()
        self.dallar = nn.ModuleList(
            [nn.Conv1d(giris_kanal, filtre, k, padding=k // 2) for k in cekirdekler]
        )
        self.bn = nn.BatchNorm1d(filtre * len(cekirdekler))

    def forward(self, x):
        return F.relu(self.bn(torch.cat([d(x) for d in self.dallar], dim=1)))


class KodlayiciBlok(nn.Module):
    def __init__(self, giris: int, cikis: int, k: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(giris, cikis, k, stride=2, padding=k // 2)
        self.bn = nn.BatchNorm1d(cikis)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class CozucuBlok(nn.Module):
    def __init__(self, giris: int, atlama: int, cikis: int, k: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(giris + atlama, cikis, k, padding=k // 2)
        self.bn = nn.BatchNorm1d(cikis)

    def forward(self, x, atlama):
        x = F.interpolate(x, size=atlama.shape[-1], mode="linear", align_corners=False)
        return F.relu(self.bn(self.conv(torch.cat([x, atlama], dim=1))))


class FazCRNN(nn.Module):
    def __init__(self, gru_gizli: int = 64, gru_katman: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cok_olcekli = CokOlcekliBlok()          # 32 kanal
        self.k1 = KodlayiciBlok(32, 48)
        self.k2 = KodlayiciBlok(48, 64)
        self.k3 = KodlayiciBlok(64, 96)
        self.gru = nn.GRU(96, gru_gizli, num_layers=gru_katman, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.c3 = CozucuBlok(2 * gru_gizli, 64, 64)
        self.c2 = CozucuBlok(64, 48, 48)
        self.c1 = CozucuBlok(48, 32, 32)
        self.cikti = nn.Conv1d(32, 3, kernel_size=1)

    def forward(self, x):                       # x: (B, 3, N)
        m = self.cok_olcekli(x)                 # (B, 32, N)
        e1 = self.k1(m)                         # (B, 48, N/2)
        e2 = self.k2(e1)                        # (B, 64, N/4)
        e3 = self.k3(e2)                        # (B, 96, N/8)
        g, _ = self.gru(e3.transpose(1, 2))     # (B, N/8, 128)
        g = g.transpose(1, 2)
        d = self.c3(g, e2)
        d = self.c2(d, e1)
        d = self.c1(d, m)
        return torch.softmax(self.cikti(d), dim=1)  # (B, 3, N): [Gürültü, P, S]


YEREL_YARI_PENCERE = 100  # örnek (±1 s)


def soft_argmax_zaman(olasilik: torch.Tensor, hz: float = ORNEKLEME_HIZI,
                      yari_pencere: int | None = YEREL_YARI_PENCERE,
                      merkez: torch.Tensor | None = None) -> torch.Tensor:
    """(B, N) olasılık eğrisinden türevlenebilir faz zamanı (saniye).

    Ağırlıklı ortalama yalnızca bir merkezin ±yari_pencere örneklik çevresinde alınır. Merkez
    verilmezse (çıkarım) olasılık tepesi (argmax) kullanılır; eğitimde gerçek etiketin indeksi
    verilir. Böylece eğitimin başında model yanlış bir tepe üretse bile zaman terimi o tepeyi
    etikete çekmeye çalışmaz. Gradyan penceredeki olasılıklardan akar.
    yari_pencere=None tüm pencereyi kullanır (eski, küresel soft-argmax; yalnızca karşılaştırma için).
    """
    n = olasilik.shape[-1]
    indeks = torch.arange(n, device=olasilik.device, dtype=olasilik.dtype)
    if yari_pencere is None:
        agirlik = olasilik / (olasilik.sum(dim=-1, keepdim=True) + 1e-8)
        return (agirlik * indeks).sum(dim=-1) / hz
    if merkez is None:
        tepe = olasilik.detach().argmax(dim=-1, keepdim=True)          # (B, 1) çıkarım
    else:
        tepe = merkez.to(olasilik.dtype).reshape(-1, 1)                 # (B, 1) eğitim
    maske = ((indeks - tepe).abs() <= yari_pencere).to(olasilik.dtype)  # (B, N)
    agirlik = olasilik * maske
    agirlik = agirlik / (agirlik.sum(dim=-1, keepdim=True) + 1e-8)
    return (agirlik * indeks).sum(dim=-1) / hz


class BilesikKayip(nn.Module):
    """P ve S fazlarını bağımsız ağırlıklarla cezalandıran bileşik kayıp fonksiyonu.

    Zaman terimi ilk `isinma_epok` epok boyunca kapalıdır (λ = 0); model önce BCE ile makul
    olasılık eğrileri öğrenir. Eğitimde yerel pencere gerçek etiketin çevresinde alınır.
    """

    def __init__(self, w_p: float = 1.0, w_s: float = 1.5, lam: float = 0.1, isinma_epok: int = 5):
        super().__init__()
        self.w_p, self.w_s, self.lam, self.isinma_epok = w_p, w_s, lam, isinma_epok

    def forward(self, tahmin, hedef, t_p=None, t_s=None, epok: int | None = None):
        # tahmin, hedef: (B, 3, N) — kanal sırası [Gürültü, P, S]
        bce = lambda i: F.binary_cross_entropy(tahmin[:, i].clamp(1e-6, 1 - 1e-6), hedef[:, i])
        kayip = bce(0) + self.w_p * bce(1) + self.w_s * bce(2)
        zaman = tahmin.new_zeros(())
        lam = 0.0 if (epok is not None and epok < self.isinma_epok) else self.lam
        if t_p is not None and lam > 0:  # t_p, t_s: (B,) saniye; NaN = pencerede faz yok
            for kanal, t in ((1, t_p), (2, t_s)):
                maske = ~torch.isnan(t)
                if maske.any():
                    t_hat = soft_argmax_zaman(tahmin[maske, kanal], merkez=t[maske] * ORNEKLEME_HIZI)
                    zaman = zaman + (t_hat - t[maske]).abs().mean()
        return kayip + lam * zaman


def gauss_etiket(n: int, faz_indeksi: float | None, sigma: int = 10) -> torch.Tensor:
    """PhaseNet'teki gibi faz anında tepe yapan Gauss etiket eğrisi."""
    if faz_indeksi is None:
        return torch.zeros(n)
    t = torch.arange(n, dtype=torch.float32)
    return torch.exp(-0.5 * ((t - faz_indeksi) / sigma) ** 2)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = FazCRNN()
    n_param = sum(p.numel() for p in model.parameters())
    x = torch.randn(4, 3, PENCERE)
    y = model(x)
    print(f"Girdi: {tuple(x.shape)}  ->  Çıktı: {tuple(y.shape)}")
    print(f"Toplam parametre: {n_param:,}")
    # Tek adımlık eğitim duman testi (sentetik etiketlerle)
    p = gauss_etiket(PENCERE, 1000.0)
    s = gauss_etiket(PENCERE, 1800.0)
    hedef = torch.stack([(1 - p - s).clamp(0, 1), p, s]).unsqueeze(0).repeat(4, 1, 1)
    kayip_fn = BilesikKayip()
    kayip = kayip_fn(y, hedef, torch.full((4,), 10.0), torch.full((4,), 18.0))
    kayip.backward()
    print(f"Bileşik kayıp (rastgele ağırlıklarla): {kayip.item():.4f} — geri yayılım OK")
