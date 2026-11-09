"""Özgün CRNN modeli ve bileşik kayıp için birim testleri."""
import math

import torch

from crnn_model import PENCERE, BilesikKayip, FazCRNN, gauss_etiket, soft_argmax_zaman


def test_cikti_boyutu_girdiyle_ayni():
    model = FazCRNN().eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, PENCERE))
    assert y.shape == (2, 3, PENCERE)


def test_cikti_her_ornekte_olasilik_dagilimi():
    model = FazCRNN().eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, PENCERE))
    assert torch.allclose(y.sum(dim=1), torch.ones(1, PENCERE), atol=1e-5)
    assert (y >= 0).all()


def test_parametre_sayisi_raporla_tutarli():
    assert sum(p.numel() for p in FazCRNN().parameters()) == 357_571


def test_farkli_pencere_uzunlugu_calisir():
    model = FazCRNN().eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 6000)).shape == (1, 3, 6000)


def test_soft_argmax_gauss_tepesini_bulur():
    etiket = gauss_etiket(PENCERE, 1000.0).unsqueeze(0)
    assert abs(soft_argmax_zaman(etiket).item() - 10.0) < 0.01  # 1000. örnek / 100 Hz


def test_bilesik_kayip_sonlu_ve_geri_yayilir():
    torch.manual_seed(0)
    model = FazCRNN()
    y = model(torch.randn(2, 3, PENCERE))
    p, s = gauss_etiket(PENCERE, 1000.0), gauss_etiket(PENCERE, 1800.0)
    hedef = torch.stack([(1 - p - s).clamp(0, 1), p, s]).unsqueeze(0).repeat(2, 1, 1)
    kayip = BilesikKayip()(y, hedef, torch.tensor([10.0, float("nan")]), torch.tensor([18.0, 18.0]))
    assert math.isfinite(kayip.item())
    kayip.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_faz_yoksa_zaman_terimi_atlanir():
    y = torch.full((1, 3, 100), 1 / 3)
    hedef = torch.zeros(1, 3, 100); hedef[:, 0] = 1
    kayip_fn = BilesikKayip(lam=1.0)
    nan = torch.tensor([float("nan")])
    assert torch.isclose(kayip_fn(y, hedef, nan, nan), kayip_fn(y, hedef))


def test_yerel_soft_argmax_taban_olasiliga_dayanikli():
    # Her örnekte 0,001 taban olasılık: küresel soft-argmax kenardaki fazı ~1,3 s içeri çeker,
    # yerel pencere bu yanlılığı 0,01 s'nin altında tutar.
    for idx in (300, 2700):
        p = gauss_etiket(PENCERE, idx, sigma=10).unsqueeze(0) + 1e-3
        assert abs(soft_argmax_zaman(p).item() - idx / 100) < 0.01
        assert abs(soft_argmax_zaman(p, yari_pencere=None).item() - idx / 100) > 1.0


def test_yerel_soft_argmax_iki_tepede_buyugu_secer():
    p = (gauss_etiket(PENCERE, 800, sigma=10) + 0.6 * gauss_etiket(PENCERE, 2200, sigma=10)).unsqueeze(0)
    assert abs(soft_argmax_zaman(p).item() - 8.0) < 0.01
    assert abs(soft_argmax_zaman(p, yari_pencere=None).item() - 8.0) > 3.0  # küresel: tepelerin arasına düşer


def test_yerel_soft_argmax_turevlenebilir():
    p = (gauss_etiket(PENCERE, 1500, sigma=10) + 1e-3).unsqueeze(0).requires_grad_(True)
    soft_argmax_zaman(p).backward()
    assert p.grad is not None and p.grad.abs().sum() > 0


def test_egitimde_pencere_etiket_cevresinde():
    # Model yanlış yerde (800. örnek) daha yüksek bir tepe üretmiş olsun; eğitimde
    # pencere gerçek etiket (2200) çevresinde alınınca küçük doğru tepe bulunur.
    p = (gauss_etiket(PENCERE, 800, sigma=10) + 0.4 * gauss_etiket(PENCERE, 2200, sigma=10)).unsqueeze(0)
    assert abs(soft_argmax_zaman(p).item() - 8.0) < 0.01                                   # çıkarım: argmax
    assert abs(soft_argmax_zaman(p, merkez=torch.tensor([2200.0])).item() - 22.0) < 0.01  # eğitim: etiket


def test_isinma_doneminde_zaman_terimi_kapali():
    kayip = BilesikKayip(isinma_epok=5)
    hedef = torch.stack([torch.zeros(PENCERE), gauss_etiket(PENCERE, 1000), gauss_etiket(PENCERE, 1600)]).unsqueeze(0)
    tahmin = torch.full((1, 3, PENCERE), 1 / 3)
    t_p, t_s = torch.tensor([10.0]), torch.tensor([16.0])
    sadece_bce = kayip(tahmin, hedef)
    assert torch.isclose(kayip(tahmin, hedef, t_p, t_s, epok=0), sadece_bce)
    assert kayip(tahmin, hedef, t_p, t_s, epok=5) > sadece_bce
