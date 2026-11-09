# Ön deneme özeti (rapor 9. ve 10. bölüm için)

> Raporda **"ön deneme"** olarak sunulmalıdır. Projenin asıl deneyleri (5 model, transfer öğrenme) henüz yapılmamıştır.

## Ne yapıldı?
Hazır (Türkiye'ye özgü veriyle eğitilmemiş) faz okuma modelleri, KOERI kayıtlarında KOERI'nin kendi bülten okumalarıyla karşılaştırıldı. Bu, BAP/2209 önerisindeki **İş Paketi 2 "Mevcut modellerin bölgesel hata analizi"nin küçük ölçekli bir provasıdır.**

| | |
|---|---|
| Olaylar | 8 deprem, 2025, Batı Türkiye, M4,9–6,2 (Marmara Denizi ×3, Sındırgı ×2, Simav ×2, Ege ×1) |
| Dalga formları | KOERI EIDA FDSN servisi, KO ağı, 47 istasyon, 3 bileşen, 18–165 km |
| Referans okumalar | ISC bülteni (ISF) üzerinden **KOERI (ISK) bülten okumaları**: 153 P (98 Pg, 55 Pn) + 43 S (34 Sg, 9 Sn) |
| Modeller | SeisBench hazır ağırlıkları: PhaseNet ve EQTransformer × {original (yazarların özgün ağırlıkları: PhaseNet Kuzey Kaliforniya, EQTransformer STEAD), stead (küresel), instance (İtalya)} |
| Ölçütler | P için ±0,5 s, S için ±1,0 s tolerans. Precision, Recall, F1, doğru okumalarda MAE ve RMSE, ortalama artık (yanlılık) |
| Eşik | P ve S olasılığı ≥ 0,3 |

**Önemli:** SeisBench'teki 17 PhaseNet ve 12 EQTransformer ağırlığının hiçbiri Türkiye'ye özgü bir veri kümesiyle eğitilmemiş. Küresel kümeler (STEAD, GEOFON, NEIC) Türkiye kayıtları içerebilir ama bölgeye odaklı değildir. Bu durum projenin problem tanımını doğrudan destekliyor.

## Sonuçlar

| Model | Faz | TP | FP | FN | Precision | Recall | F1 | MAE (s) | RMSE (s) | Ort. artık (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| PhaseNet-original | P | 123 | 37 | 30 | 0,77 | 0,80 | 0,79 | 0,18 | 0,23 | +0,06 |
| PhaseNet-stead | P | 116 | 26 | 37 | 0,82 | 0,76 | 0,79 | 0,16 | 0,21 | +0,02 |
| PhaseNet-instance | P | 127 | 27 | 26 | 0,82 | 0,83 | **0,83** | 0,15 | 0,21 | +0,01 |
| EQT-original | P | 65 | 15 | 88 | 0,81 | 0,42 | 0,56 | 0,16 | 0,20 | −0,00 |
| EQT-stead | P | 115 | 30 | 38 | 0,79 | 0,75 | 0,77 | 0,17 | 0,22 | +0,04 |
| EQT-instance | P | 127 | 22 | 26 | 0,85 | 0,83 | **0,84** | 0,18 | 0,23 | +0,03 |
| PhaseNet-original | S | 27 | 8 | 16 | 0,77 | 0,63 | 0,69 | 0,31 | 0,43 | +0,23 |
| PhaseNet-stead | S | 31 | 13 | 12 | 0,70 | 0,72 | **0,71** | 0,32 | 0,41 | +0,15 |
| PhaseNet-instance | S | 16 | 2 | 27 | 0,89 | 0,37 | 0,52 | 0,34 | 0,45 | +0,19 |
| EQT-original | S | 20 | 4 | 23 | 0,83 | 0,47 | 0,60 | 0,33 | 0,44 | +0,20 |
| EQT-stead | S | 30 | 11 | 13 | 0,73 | 0,70 | **0,71** | 0,29 | 0,39 | +0,16 |
| EQT-instance | S | 19 | 4 | 24 | 0,83 | 0,44 | 0,58 | 0,34 | 0,44 | +0,20 |

Şekiller: `sekiller/on_deneme_f1.png`, `sekiller/on_deneme_artiklar.png`, `sekiller/on_deneme_ornek_kayit.png`

## Yorum (rapora taslak cümleler)
1. **P fazı:** En iyi F1 0,84'tür (EQT-instance). Doğru okunan P fazlarında zamanlama hatası küçüktür (MAE ≈ 0,15–0,18 s) ve yanlılık yoktur. Ancak P okumalarının yaklaşık %17'si tolerans dışında kalmakta ya da hiç okunamamaktadır.
2. **S fazı:** F1 en fazla 0,71'dir. Modeller S'yi KOERI okumasına göre ortalamada 0,15–0,23 s geç okumaktadır. **Ancak olay bazlı analiz (9 Kasım 2026, `src/olay_analizi.py`) bu ortalamanın yanıltıcı olduğunu gösterdi:** gecikme üç Marmara depreminde belirgin (model ortalamaları +0,30…+0,45 s), diğer beş depremde belirsizdir (−0,09…+0,10 s). S okumalarının 21/43'ü tek depremden gelir; olay düzeyinde %95 güven aralığı 6 modelin 3'ünde sıfırı içerir. Sn/Sg karışması açıklamıyor (gecikme Sg'de). Bulgu bu nedenle bileşik kaybın kanıtı değil, asıl deneyde sınanacak bir hipotezdir: gecikmeyi KOERI etiketleriyle ince ayar tek başına giderebilir; bileşik kaybın ek katkısı ablasyonla ölçülecektir.
3. **Eğitim verisinin bölgesi önemli:** İtalya (INSTANCE) ağırlıkları P fazında en iyi sonucu verirken S fazında recall'u düşüktür (0,37–0,44). Küresel STEAD ağırlıkları S fazında daha dengelidir. Tek bir hazır ağırlık iki fazda birden en iyi değildir. Bu da Türkiye'ye özgü ince ayarın gerekçesidir.
4. **EQT-original** bu eşikte çok temkinlidir (P recall 0,42). Bu, eşik seçiminin de değerlendirme tasarımına dahil edilmesi gerektiğini gösterir.

## Sınırlılıklar (raporda mutlaka yazılmalı)
- Örneklem küçüktür (8 olay, 196 referans okuma). İstatistiksel anlamlılık testi yapılmamıştır.
- Referans okumalar KOERI'nin ISC'ye raporladığı bülten okumalarıdır. Elle/otomatik ayrımı bültende belirtilmez ve çözünürlük 0,1 s'dir.
- S referansı azdır (43). Bu yüzden S sonuçlarının belirsizliği yüksektir.
- FP yalnızca KOERI'nin o fazı okuduğu istasyonlarda sayılmıştır. Büyük olaylarda analiz penceresine artçılar girebilir.
- Tüm ağırlıklara ortak 0,3 eşiği uygulanmıştır; SeisBench ağırlıklarının kendi optimize edilmiş varsayılan eşikleri kullanılmamıştır. EQT-original'ın düşük recall'u kısmen bundandır.
- Ortalama artık ve MAE yalnızca tolerans içindeki okumalardan hesaplanmıştır.
- Literatür notu: Münchmeyer vd. (2022) bölgeler arası aktarımda kaybın genelde **hafif** olduğunu bulmuştur. Bu yüzden bulgular "modeller Türkiye'de çalışmıyor" diye yorumlanmamalıdır; S gecikmesi de olay bazında Marmara'ya özgü görünmektedir.

## Hız ölçümü (AMD Ryzen AI 7 350 dizüstü CPU, WSL2, 3 iş parçacığı, GPU kullanılmadan, eğitimsiz ağırlıklarla)
| Model | Pencere | Parametre | 1 pencere (ms) | 32 pencere (ms) |
|---|---|---|---|---|
| Özgün CRNN | 30 s | 357.571 | 16,6 | 141,0 |
| PhaseNet | 30 s | 268.443 | 1,0 | 11,0 |
| EQTransformer | 60 s | 376.935 | 9,1 | 76,1 |

Yorum: 30 s'lik kaydı CRNN 17 ms'de işliyor, yani gerçek zamandan yaklaşık 1800 kat hızlı. Bu, işlevsel olmayan gecikme gereksinimini (ör. pencere başına < 100 ms) karşılıyor. Ancak CRNN, PhaseNet'ten yaklaşık 16 kat yavaş. Fark profillenmedi; muhtemel nedenler tam çözünürlükteki büyük çekirdekli (k=63) evrişimler ve GRU katmanıdır. Bu durum raporda "optimizasyon gereksinimi" olarak belirtilebilir.
