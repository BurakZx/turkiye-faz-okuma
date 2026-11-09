"""Dikey dilimin uçtan uca demo videosunu altyazılı olarak kaydeder (canlı demo aksarsa yedek).

Akış: (1) API belgeleri sayfasından /pick ile bir KOERI kaydı yüklenir → (2) panelde inceleme
kuyruğu → (3) bir okuma onaylanır → (4) bir okuma düzeltilip kaydedilir → (5) /egitim-etiketleri
çıktısında onaylanan ve düzeltilen okumalar görünür.

Ön koşul: API (8000) ve panel (8501) çalışıyor, kuyrukta okuma var; ffmpeg ve DejaVu Sans kurulu.
Çalıştırma:  .venv/bin/python -m servis.demo_video [yüklenecek.mseed]
"""
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

KOK = Path(__file__).resolve().parents[1]
CIKTI = KOK / "sekiller" / "dikey_dilim_demo.mp4"
API, PANEL = "http://127.0.0.1:8000", "http://127.0.0.1:8501"
YAZI_TIPI = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
YUKLENECEK = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((KOK / "veri" / "mseed").glob("us7000pufs_*.mseed"))[9]


def kaydet():
    altyazi = []  # (başlangıç_s, metin)
    with tempfile.TemporaryDirectory() as gecici, sync_playwright() as p:
        tarayici = p.chromium.launch()
        baglam = tarayici.new_context(viewport={"width": 1280, "height": 800},
                                      record_video_dir=gecici, record_video_size={"width": 1280, "height": 800})
        sayfa = baglam.new_page()
        t0 = time.monotonic()
        not_ = lambda m: altyazi.append((time.monotonic() - t0, m))

        # 1) API: /pick ile kayıt yükleme
        sayfa.goto(f"{API}/docs", wait_until="networkidle")
        not_("1) API: /pick uç noktasına bir KOERI kaydı (MiniSEED) yükleniyor")
        sayfa.wait_for_timeout(1500)
        sayfa.get_by_text("/pick", exact=True).first.click()
        sayfa.wait_for_timeout(1200)
        sayfa.get_by_role("button", name="Try it out").click()
        sayfa.locator("input[type=file]").set_input_files(str(YUKLENECEK))
        sayfa.wait_for_timeout(1000)
        sayfa.get_by_role("button", name="Execute").click()
        sayfa.wait_for_selector(".responses-wrapper .response-col_description pre", timeout=60000)
        sayfa.locator(".responses-wrapper .live-responses-table").first.scroll_into_view_if_needed()
        not_("Model okumaları kaydetti; düşük güvenli olanlar inceleme kuyruğuna düştü")
        sayfa.wait_for_timeout(4500)

        # 2) Panel
        not_("2) Analist paneli: inceleme kuyruğu, dalga formu, model ve KOERI okuması")
        sayfa.goto(PANEL, wait_until="networkidle")
        sayfa.wait_for_selector("text=Düzelt ve kaydet", timeout=60000)
        sayfa.wait_for_timeout(3500)

        # 3) Onayla
        sayfa.locator('div[role="radiogroup"] label').nth(1).click()
        not_("3) Analist bir okumayı seçip onaylıyor")
        sayfa.wait_for_timeout(2500)
        sayfa.get_by_role("button", name="Onayla").click()
        sayfa.wait_for_timeout(3500)

        # 4) Düzelt
        sayfa.mouse.wheel(0, 450)
        not_("4) Analist bir okumayı düzeltip kaydediyor")
        sayfa.wait_for_timeout(2000)
        sayfa.get_by_role("button", name="Düzelt ve kaydet").click()
        sayfa.wait_for_timeout(4000)
        sayfa.mouse.wheel(0, -450)
        not_("Sayaçlar güncellendi: bekleyen azaldı, onaylanan/düzeltilen arttı")
        sayfa.wait_for_timeout(3500)

        # 5) Eğitim etiketleri
        not_("5) Onaylanan ve düzeltilen okumalar /egitim-etiketleri ile yeniden eğitime gidiyor")
        sayfa.goto(f"{API}/egitim-etiketleri")
        try:
            sayfa.get_by_text("Pretty-print").click(timeout=3000)  # Chromium JSON görünümü
        except Exception:
            pass
        sayfa.add_style_tag(content="body, pre { font-size: 17px !important; white-space: pre-wrap; }")
        sayfa.wait_for_timeout(5500)
        sure = time.monotonic() - t0
        video = sayfa.video.path()
        baglam.close()
        tarayici.close()

        webm = Path(gecici) / "demo.webm"
        shutil.copy(video, webm)
        filtre = []
        for i, (bas, metin) in enumerate(altyazi):
            son = altyazi[i + 1][0] if i + 1 < len(altyazi) else sure
            dosya = Path(gecici) / f"a{i}.txt"
            dosya.write_text(metin, encoding="utf-8")
            filtre.append(f"drawtext=fontfile={YAZI_TIPI}:textfile={dosya}:fontsize=24:fontcolor=white:"
                          f"box=1:boxcolor=0x1F4E79@0.85:boxborderw=12:x=(w-text_w)/2:y=h-th-28:"
                          f"enable='between(t,{bas:.2f},{son:.2f})'")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-vf", ",".join(filtre),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                        str(CIKTI)], check=True)
    print("Kaydedildi:", CIKTI, f"({sure:.1f} s)")


if __name__ == "__main__":
    kaydet()
