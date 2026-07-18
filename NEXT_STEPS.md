# Eğitim Sonrası Yapılacaklar (Projeyi "iyi"den "çok iyi"ye taşıyan kısım)

> Model eğitilip `outputs/qlora-checkpoints/final-adapter/` oluştuktan sonra bu listeyi uygula.
> Projenin değeri buradan sonra ortaya çıkıyor: **gerçek sonuçlar + savunabilmek.**

## 1. Değerlendirmeyi çalıştır (EN ÖNEMLİ)
Base model vs fine-tuned model karşılaştırması — BLEU, ROUGE-L, BERTScore (+ istersen LLM-judge).

```bash
pip install -q sacrebleu rouge-score bert-score
# LLM-judge'suz (API anahtarı gerekmez):
python src/evaluate.py --config configs/config.yaml \
    --adapter outputs/qlora-checkpoints/final-adapter \
    --test data/splits/test.jsonl --skip-judge
# Gemini anahtarın varsa LLM-judge dahil:
python src/evaluate.py --config configs/config.yaml \
    --adapter outputs/qlora-checkpoints/final-adapter \
    --test data/splits/test.jsonl --api-key $GEMINI_API_KEY
```
Sonuç: `outputs/evaluation/evaluation_report.json`. Bu sayıları README + MODEL_CARD'a işle.

## 2. README'ye 3–5 somut örnek çıktı ekle
Aynı soruya **base model** ve **fine-tuned model** ne cevap veriyor, yan yana göster.
(Notebook'taki 7. hücre `test_inference` fine-tuned çıktıları verir; base için adapter'sız çalıştır.)

## 3. Limitasyonları dürüstçe yaz
- Küçük veri (1.000), çoğu sentetik (861 LLM üretimi) → genelleme sınırı.
- 1.000 örnekle 8B fine-tune → overfitting riski; kazanç ölçülmeli.
- Alan dışı (K-12 STEM dışı) sorularda güvenilmez.
Sınırlarını bilmek olgunluk göstergesidir; mülakatçı bunu sever.

## 4. Her teknik seçimi 2 cümleyle açıklayabil (mülakat hazırlığı)
Kendine sor, cevaplayabildiğinden emin ol:
- Neden **QLoRA / 4-bit**? (bellek: 8B'yi tek T4/L4'e sığdırmak)
- Neden **LoRA tüm linear katmanlara** (q/v yerine)? (QLoRA makalesi: kaliteyi en çok bu artırır)
- **%35 red oranı** neye göre? (HITL filtreleme: uzunluk/biçim/dil/benzerlik + manuel inceleme)
- **NEFTune** ne yapar? (embedding'e gürültü → daha iyi genelleme)
- **Self-Instruct** nedir? (seed'lerden LLM ile veri genişletme, Wang vd. 2023)
- **BLEU vs BERTScore** farkı? (yüzeysel örtüşme vs anlamsal benzerlik)
- **LLM-as-judge** neden? (otomatik metrikler açık uçlu cevaplarda yetersiz)

## 5. (Opsiyonel) HuggingFace Hub'a yayınla
```python
from huggingface_hub import login; login()   # write yetkili token
```
```bash
python src/publish_hub.py --config configs/config.yaml --hf-token $HF_TOKEN
```
`configs/config.yaml` içindeki `hub.dataset_repo` / `hub.model_repo` adlarını kendi kullanıcı adınla güncelle.

---
**Özet hedef:** "fine-tune ettim" değil, **"fine-tune ettim, baseline'a göre şu kadar iyileşti, şu örneklerde şöyle daha iyi, şu sınırları var"** diyebilmek.
