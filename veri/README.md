# veri/

Bu klasör git'e eklenmez. `src/on_deneme.py` çalıştırıldığında otomatik dolar:

| Alt klasör / dosya | İçerik | Kaynak |
|---|---|---|
| `isc/*.isf.html` | Olay başına ISC bülteni (ISF2), faz okumaları ve raporlayan ajans | https://www.isc.ac.uk (ISC Bulletin, doi:10.31905/D808B830) |
| `mseed/*.mseed` | KO ağı istasyonlarının 3 bileşenli dalga formları (orijin −5 s / +90 s) | KOERI EIDA FDSN, http://eida-service.koeri.boun.edu.tr/fdsnws/ (doi:10.7914/SN/KO) |
| `ko_istasyonlar.txt` | KO ağı istasyon listesi | KOERI FDSN station servisi |

**Not:** ISC'nin 2024/11 sonrası bülteni henüz gözden geçirilmemiştir ve ileride değişebilir. Ön denemede kullanılan referans okumaların 8 Kasım 2026 tarihli anlık görüntüsü `sonuclar/referans_okumalar.csv` dosyasında saklanmaktadır.
