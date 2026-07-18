"""
HuggingFace Hub Yayınlama
==========================
Veri seti ve adapter ağırlıklarını HuggingFace Hub'a push eder.
Model card ve dataset card otomatik oluşturur.

Kullanım:
    python src/publish_hub.py --config configs/config.yaml --hf-token YOUR_TOKEN
"""

import os
import json
import argparse
from pathlib import Path

import yaml
import jsonlines
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file
from datasets import Dataset, DatasetDict

# ─── Config ─────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Dataset Card ──────────────────────────────────────────

def generate_dataset_card(config: dict, stats: dict) -> str:
    """HuggingFace dataset card (README.md) oluştur."""
    return f"""---
language:
  - tr
license: apache-2.0
task_categories:
  - text-generation
  - question-answering
tags:
  - stem-education
  - turkish
  - k12
  - arduino
  - scratch
  - mblock
  - robotics
  - coding-education
  - instruction-tuning
size_categories:
  - n<1K
---

# Eding STEM TR Instruct 1K

Türkçe K-12 STEM ve kodlama eğitimi için instruction-tuning veri seti.

## Veri Seti Hakkında

Bu veri seti, Türkiye'deki K-12 seviyesinde STEM ve kodlama eğitimi için
hazırlanmış 1.000 instruction-output çiftinden oluşur.

### Kategoriler
- **Arduino**: LED, sensör, motor projeleri, devre tasarımı
- **Scratch**: Blok tabanlı programlama, oyun yapımı, animasyon
- **mBlock**: mBot robot programlama, sensör kullanımı
- **Robotik**: PID kontrol, çizgi izleme, engelden kaçma
- **Python STEM**: Algoritmalar, veri yapıları, bilimsel hesaplama
- **Elektronik**: Devre analizi, breadboard, temel elektronik
- **Algoritma**: Sıralama, arama, karmaşıklık analizi

### Zorluk Seviyeleri
- **İlkokul** (1-4. sınıf)
- **Ortaokul** (5-8. sınıf)
- **Lise** (9-12. sınıf)

## Üretim Metodolojisi

Hibrit pipeline ile derlenmiştir:
1. **Manuel Seed Yazımı**: ~{stats.get('total_seeds', 150)} uzman tarafından yazılmış örnek
2. **Self-Instruct Genişletme**: LLM ile ~{stats.get('llm_generated', 850)} yeni örnek
3. **Human-in-the-Loop Filtreleme**: ~%{int(stats.get('rejection_rate', 0.35) * 100)} red oranı ile kalite kontrolü

## Veri Formatı

```json
{{
  "instruction": "Arduino ile LED yakıp söndüren kodu yaz.",
  "input": "",
  "output": "Detaylı açıklama ve kod...",
  "category": "arduino",
  "difficulty": "ortaokul"
}}
```

## Splits

| Split | Örnek Sayısı |
|-------|-------------|
| Train | {stats.get('train_count', '~850')} |
| Validation | {stats.get('val_count', '~50')} |
| Test | {stats.get('test_count', '~100')} |

## Kullanım

```python
from datasets import load_dataset

dataset = load_dataset("{config['hub']['dataset_repo']}")
print(dataset["train"][0])
```

## Lisans

Apache 2.0

## Atıf

```bibtex
@misc{{eding-stem-tr-2026,
  title={{Eding STEM TR Instruct 1K: Türkçe K-12 STEM Eğitimi Veri Seti}},
  author={{Kacar, Alim}},
  year={{2026}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/datasets/{config['hub']['dataset_repo']}}}
}}
```
"""


# ─── Model Card ─────────────────────────────────────────────

def generate_model_card(config: dict, eval_report: dict = None) -> str:
    """HuggingFace model card oluştur."""
    lora = config["training"]["lora"]
    args = config["training"]["args"]

    eval_section = ""
    if eval_report:
        ft_metrics = eval_report.get("model_comparison", {}).get("finetuned", {}).get("automatic_metrics", {})
        bl_metrics = eval_report.get("model_comparison", {}).get("baseline_zero_shot", {}).get("automatic_metrics", {})
        if ft_metrics:
            eval_section = f"""
## Değerlendirme Sonuçları

| Metrik | Fine-tuned | Baseline (Zero-shot) | İyileşme |
|--------|-----------|---------------------|----------|
| BLEU | {ft_metrics.get('bleu', 'N/A')} | {bl_metrics.get('bleu', 'N/A')} | {eval_report.get('improvements', {}).get('bleu', {}).get('improvement_pct', 'N/A')}% |
| ROUGE-L | {ft_metrics.get('rougeL', 'N/A')} | {bl_metrics.get('rougeL', 'N/A')} | {eval_report.get('improvements', {}).get('rougeL', {}).get('improvement_pct', 'N/A')}% |
| BERTScore F1 | {ft_metrics.get('bertscore_f1', 'N/A')} | {bl_metrics.get('bertscore_f1', 'N/A')} | {eval_report.get('improvements', {}).get('bertscore_f1', {}).get('improvement_pct', 'N/A')}% |
"""

    return f"""---
language:
  - tr
license: apache-2.0
library_name: peft
base_model: {config['training']['base_model']}
tags:
  - qlora
  - stem-education
  - turkish
  - instruction-tuning
  - k12
pipeline_tag: text-generation
datasets:
  - {config['hub']['dataset_repo']}
---

# Turkish Llama 8B - STEM QLoRA Adapter

{config['training']['base_model']} üzerine QLoRA ile fine-tune edilmiş
Türkçe K-12 STEM/kodlama eğitimi adapter'ı.

## Model Detayları

- **Base Model**: `{config['training']['base_model']}`
- **Fine-tuning Yöntemi**: QLoRA (4-bit NF4 quantization)
- **LoRA Rank**: {lora['r']}
- **LoRA Alpha**: {lora['lora_alpha']}
- **Target Modules**: {', '.join(lora['target_modules'])}
- **Eğitim Epoch**: {args['num_train_epochs']}
- **Learning Rate**: {args['learning_rate']}
- **Eğitim Verisi**: [{config['hub']['dataset_repo']}](https://huggingface.co/datasets/{config['hub']['dataset_repo']})

{eval_section}

## Kullanım

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Base model + adapter yükle
base_model = AutoModelForCausalLM.from_pretrained(
    "{config['training']['base_model']}",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "{config['hub']['model_repo']}")
tokenizer = AutoTokenizer.from_pretrained("{config['hub']['model_repo']}")

# Inference
prompt = "Arduino ile servo motor nasıl kontrol edilir?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Lisans

Apache 2.0
"""


# ─── Yayınlama ──────────────────────────────────────────────

def publish_dataset(
    config: dict,
    data_dir: str = "data/splits",
    hf_token: str = None,
):
    """Veri setini HuggingFace Hub'a yükle."""
    repo_id = config["hub"]["dataset_repo"]
    api = HfApi(token=hf_token)

    print(f"\n[HUB] Dataset yükleniyor: {repo_id}")

    # Repo oluştur
    create_repo(
        repo_id,
        repo_type="dataset",
        private=config["hub"]["private"],
        token=hf_token,
        exist_ok=True,
    )

    # JSONL → HF Dataset formatı
    #   dosya adları: train.jsonl / validation.jsonl / test.jsonl
    file_map = [("train", "train"), ("validation", "validation"), ("test", "test")]
    splits = {}
    for key, fname in file_map:
        path = os.path.join(data_dir, f"{fname}.jsonl")
        if os.path.exists(path):
            with jsonlines.open(path) as reader:
                data = list(reader)
            splits[key] = Dataset.from_list(data)
            print(f"  {key}: {len(data)} örnek")

    if not splits:
        raise ValueError(f"Veri bulunamadı: {data_dir}")

    dataset_dict = DatasetDict(splits)

    # Push
    dataset_dict.push_to_hub(repo_id, token=hf_token)

    # İstatistikleri topla (data/dataset_stats.json)
    stats_path = os.path.join(os.path.dirname(data_dir) or "data", "dataset_stats.json")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)

    stats["train_count"] = len(splits.get("train", []))
    stats["val_count"] = len(splits.get("validation", []))
    stats["test_count"] = len(splits.get("test", []))

    # Dataset card yükle
    card = generate_dataset_card(config, stats)
    card_path = "/tmp/dataset_readme.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)

    upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=hf_token,
    )

    print(f"[OK] Dataset yüklendi: https://huggingface.co/datasets/{repo_id}")


def publish_model(
    config: dict,
    adapter_path: str,
    eval_report_path: str = None,
    hf_token: str = None,
):
    """Model adapter'ını HuggingFace Hub'a yükle."""
    repo_id = config["hub"]["model_repo"]

    print(f"\n[HUB] Model yükleniyor: {repo_id}")

    # Repo oluştur
    create_repo(
        repo_id,
        repo_type="model",
        private=config["hub"]["private"],
        token=hf_token,
        exist_ok=True,
    )

    # Eval report yükle
    eval_report = {}
    if eval_report_path and os.path.exists(eval_report_path):
        with open(eval_report_path) as f:
            eval_report = json.load(f)

    # Model card
    card = generate_model_card(config, eval_report)
    card_path = os.path.join(adapter_path, "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)

    # Adapter dosyalarını yükle
    upload_folder(
        folder_path=adapter_path,
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
    )

    print(f"[OK] Model yüklendi: https://huggingface.co/{repo_id}")


# ─── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuggingFace Hub Yayınlama")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--hf-token", default=None, help="HF token (veya HF_TOKEN env var)")
    parser.add_argument("--data-dir", default="data/splits")
    parser.add_argument("--adapter", default="outputs/qlora-checkpoints/final-adapter")
    parser.add_argument("--eval-report", default="outputs/evaluation/evaluation_report.json")
    parser.add_argument(
        "--what",
        choices=["dataset", "model", "both"],
        default="both",
        help="Ne yüklenecek",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF token gerekli: --hf-token veya HF_TOKEN env var")

    if args.what in ("dataset", "both"):
        publish_dataset(config, args.data_dir, hf_token)

    if args.what in ("model", "both"):
        publish_model(config, args.adapter, args.eval_report, hf_token)
