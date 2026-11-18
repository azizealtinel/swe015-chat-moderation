# SWE015 – Gerçek Zamanlı Türkçe Chat Moderasyonu (Use Case 4)

Bir oyun platformundaki sohbet mesajlarını **CPU üzerinde, mesaj başına ≤ 50 ms** içinde
*temiz / saldırgan* olarak etiketleyen ve saldırgan mesajları gizleyen uygulama.

Aynı test seti (OffensEval-TR 2020, 3.528 tweet) üzerinde dört model karşılaştırılır:

| # | Model | Alan |
|---|-------|------|
| 0 | Sözlük + minimum edit distance (kural tabanlı ek baseline) | B |
| 1 | tf-idf (kelime + karakter n-gram) + lojistik regresyon — **klasik baseline** | B |
| 2 | Kendi eğittiğimiz word2vec (skip-gram, negative sampling) + feedforward ağ (SGD + backprop) | A |
| 3 | BERTurk (`dbmdz/bert-base-turkish-cased`) fine-tune — **Transformer** | C |
| 4 | BERTurk, dinamik INT8 quantization — **uygulamada çalışan model** | D |

Tüm ön işleme (`src/textproc.py`): regex ile leetspeak çözme (`s4n4 → sana`), harf tekrarlarını
indirgeme (`çoooook → çok`), Türkçe karakter birleştirme (`çok = cok`).

## Kurulum

```bash
git clone https://github.com/azizealtinel/swe015-chat-moderation.git
cd swe015-chat-moderation
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma (tek komut)

```bash
python app.py
```

Model ağırlıkları repoda değildir; ilk çalıştırmada herkese açık Google Drive klasöründen
([model klasörü](https://drive.google.com/drive/folders/1qXhEniBflZYqNKvOaRUH1Sx1tRrpFTU5)) `models/` klasörüne otomatik indirilir (~630 MB).
Tarayıcıda `http://127.0.0.1:7860` açılır. Uygulama varsayılan olarak quantized (INT8) BERTurk'ü CPU'da çalıştırır.

Colab'da herkese açık link için: `python app.py --share` · FP32 modeli de karşılaştırmaya eklemek için: `--fp32`

## Modelleri yeniden eğitmek

`notebooks/train_all.ipynb` dosyasını Google Colab'da (T4 GPU) açıp tüm hücreleri çalıştırın.
Veri seti Hugging Face Hub'dan otomatik indirilir. Sonuçlar (`macro-F1`, boyut, gecikme)
`results.json` dosyasına yazılır.

## Sonuçlar

Tablo `results.json` dosyasından alınmıştır. Gecikme: tek mesaj, CPU, 200 (model 0–2) / 100 (model 3–4) çalıştırmanın ortalaması.
Donanım: Google Colab, x86_64 CPU, 2 çekirdek.

| Model | Alan | Macro-F1 | F1 (offensive) | Boyut (MB) | Gecikme (ms) |
|---|---|---|---|---|---|
| 0 · Sözlük + edit distance | B | 0.572 | 0.426 | < 0.01 | 5.24 |
| 1 · tf-idf + lojistik regresyon | B | 0.770 | 0.634 | 3.50 | 3.03 |
| 2 · word2vec + feedforward ağ | A | 0.716 | 0.558 | 7.79 | 0.32 |
| 3 · BERTurk FP32 | C | 0.814 | 0.706 | 442.50 | 129.83 |
| **4 · BERTurk INT8 (uygulamada)** | **D** | **0.816** | **0.706** | **186.04** | **43.34** |

Gecikme bütçesi: CPU'da mesaj başına 50 ms → INT8 model 43.3 ms ile bütçe içinde.
Test tahminleri: model 3 T4 GPU'da, model 4 CPU'da hesaplandı. Ham sayılar: `results.json`.

## Veri ve kaynaklar

- Veri: OffensEval-TR 2020 — Çöltekin, Ç. (2020). *A Corpus of Turkish Offensive Language on Social Media.* LREC 2020.
  Train/dev bölmesi: [`stefan-it/offenseval2020_tr`](https://huggingface.co/datasets/stefan-it/offenseval2020_tr) (CC BY 2.0).
- Ön eğitimli model: BERTurk — [`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased).

## Repo yapısı

```
app.py                 Gradio arayüzü (giriş noktası)
src/textproc.py        normalizasyon, edit distance, sözlük eşleştirme
src/models.py          modellerin yüklenmesi ve tahmin
notebooks/train_all.ipynb  veri, eğitim, quantization, ölçümler
results.json           ölçülen sonuçlar
requirements.txt
```
