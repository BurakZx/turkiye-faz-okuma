"""Dikey dilimin kısa demo videosunu kaydeder (sunumda canlı demo aksarsa yedek olarak).

Akış: panel açılır → bir okuma seçilir → onaylanır → sonraki okuma KOERI'ye göre düzeltilir.
Ön koşul: API (8000) ve panel (8501) çalışıyor olmalı; ffmpeg kurulu olmalı.
Çalıştırma:  .venv/bin/python -m servis.demo_video
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

CIKTI = Path(__file__).resolve().parents[1] / "sekiller" / "dikey_dilim_demo.mp4"

with tempfile.TemporaryDirectory() as gecici, sync_playwright() as p:
    tarayici = p.chromium.launch()
    baglam = tarayici.new_context(viewport={"width": 1280, "height": 860},
                                  record_video_dir=gecici, record_video_size={"width": 1280, "height": 860})
    sayfa = baglam.new_page()
    sayfa.goto("http://127.0.0.1:8501", wait_until="networkidle")
    sayfa.wait_for_selector("text=Düzelt ve kaydet", timeout=60000)
    sayfa.wait_for_timeout(3000)
    sayfa.locator('div[role="radiogroup"] label').nth(1).click()  # ikinci okumayı seç
    sayfa.wait_for_timeout(3500)
    sayfa.get_by_role("button", name="Onayla").click()             # onayla
    sayfa.wait_for_timeout(4000)
    sayfa.mouse.wheel(0, 400)                                      # düzeltme alanına in
    sayfa.wait_for_timeout(2000)
    sayfa.get_by_role("button", name="Düzelt ve kaydet").click()   # KOERI'ye göre düzelt
    sayfa.wait_for_timeout(4000)
    sayfa.mouse.wheel(0, -400)
    sayfa.wait_for_timeout(3000)
    video = sayfa.video.path()
    baglam.close()
    tarayici.close()
    webm = Path(gecici) / "demo.webm"
    shutil.copy(video, webm)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-crf", "26", str(CIKTI)], check=True)
print("Kaydedildi:", CIKTI)
