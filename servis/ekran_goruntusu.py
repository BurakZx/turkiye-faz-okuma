"""Çalışan analist panelinin ekran görüntüsünü alır (rapor ve sunum için).

Ön koşul: API (8000) ve panel (8501) çalışıyor olmalı.
Çalıştırma:  .venv/bin/python -m servis.ekran_goruntusu
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

CIKTI = Path(__file__).resolve().parents[1] / "sekiller" / "dikey_dilim_panel.png"

with sync_playwright() as p:
    tarayici = p.chromium.launch()
    sayfa = tarayici.new_page(viewport={"width": 1440, "height": 1180}, device_scale_factor=1.5)
    sayfa.goto("http://127.0.0.1:8501", wait_until="networkidle")
    sayfa.wait_for_selector("text=Düzelt ve kaydet", timeout=60000)
    sayfa.wait_for_timeout(2500)
    sayfa.screenshot(path=str(CIKTI), full_page=True)
    tarayici.close()

from PIL import Image  # noqa: E402  boş üst/alt kenarları kırp
g = Image.open(CIKTI)
g.crop((0, int(g.height * 0.07), g.width, int(g.height * 0.85))).save(CIKTI)
print("Kaydedildi:", CIKTI)
