# Fake News Detection with BERT & Flask

Bu proje, BERT tabanlı bir modelle haberlerin sahte (fake) mi yoksa gerçek (real) mi olduğunu tahmin eden bir yapay zeka sistemi içerir. Model Flask API aracılığıyla kullanılabilir hale getirilmiştir.

---

## Proje Yapısı

```
fake_news_bot/
├── Fake.csv                  # Sahte haber veri seti
├── True.csv                  # Gerçek haber veri seti
├── train_bert_model.py       # BERT modelini eğiten Python dosyası
├── app.py                    # Flask API dosyası
├── bert_model/               # Eğitilen model dosyaları
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.txt
```

---

## Kurulum Adımları

```bash
# Sanal ortam oluştur ve aktive et
python3 -m venv venv
source venv/bin/activate

# Gerekli paketleri yükle
pip install -r requirements.txt
```

---

## Dataseti 

## Modeli Eğitme

```bash
python train_bert_model.py
```
> Bu adım sonunda `bert_model/` klasörü oluşur ve eğitimli model kaydedilir.

---

## Flask API'yi Başlatma

```bash
python app.py
```
> API `localhost:5001/predict` adresinde çalışmaya başlar.

---

## API Kullanımı

### POST /predict

**Girdi:**
```json
{
  "text": "The government confirmed the launch of a new satellite."
}
```

**Çıktı:**
```json
{
  "prediction": "REAL"
}
```

### Örnek curl komutu:
```bash
curl -X POST http://127.0.0.1:5001/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "The government confirmed the launch of a new satellite."}'
```

---

## Kullanılan Kütüphaneler
- Transformers (HuggingFace)
- Datasets
- PyTorch
- Flask
- scikit-learn
- pandas

---

## Geliştirici Notları
- Eğitim süresi dataset büyüklüğüne göre değişir (3–10 dakika).
- `bert_model/` klasörü silinirse yeniden eğitim yapılması gerekir.
- Eğitim sürecindeki `loss` değerleri ve `confusion matrix` görselleri rapor için kullanılabilir.

---

## Sonuç
Bu sistem, haber içeriklerinin dil kalıplarına ve bağlamsal yapısına göre gerçek/sahte ayrımını yüksek doğrulukla tahmin edebilen modern bir NLP uygulamasıdır.
