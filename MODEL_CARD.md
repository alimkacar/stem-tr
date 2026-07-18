---
base_model: ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1
library_name: peft
license: llama3
language:
- tr
tags:
- qlora
- lora
- peft
- stem
- education
- k12
- turkish
- text-generation
pipeline_tag: text-generation
datasets:
- sehinsahfanboy/stem-tr-instruct-1k
metrics:
- bleu
- rouge
- bertscore
model-index:
- name: Turkish-Llama-8B-STEM-QLoRA
  results:
  - task:
      type: text-generation
      name: Turkish STEM Instruction Following
    dataset:
      name: stem-tr-instruct-1k (test split)
      type: sehinsahfanboy/stem-tr-instruct-1k
    metrics:
    - type: bleu
      value: 46.94
      name: BLEU
    - type: rouge
      value: 61.38
      name: ROUGE-L
    - type: bertscore
      value: 81.43
      name: BERTScore-F1
---

<div align="center">

# 🧠 Turkish-Llama-8B-STEM-QLoRA

### A QLoRA adapter for **Turkish K–12 STEM & coding** instruction following

![Base](https://img.shields.io/badge/base-Turkish--Llama--8B-blue?style=for-the-badge)
![QLoRA](https://img.shields.io/badge/QLoRA-4--bit%20NF4-6E56CF?style=for-the-badge)
![PEFT](https://img.shields.io/badge/%F0%9F%A4%97-PEFT%20%2F%20LoRA-FFD21E?style=for-the-badge)
![Lang](https://img.shields.io/badge/language-Turkish-E30A17?style=for-the-badge)

</div>

---

A **LoRA adapter** fine-tuned with **QLoRA** on top of [`ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1`](https://huggingface.co/ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1), specialised for **K–12 STEM and coding education in Turkish** (Arduino, Scratch, mBlock, robotics, Python, electronics, algorithms). Trained on the [`stem-tr-instruct-1k`](https://huggingface.co/datasets/sehinsahfanboy/stem-tr-instruct-1k) dataset.

## 📊 Evaluation

On a held-out test set (100 examples), the fine-tuned model **substantially beats** the zero-shot base model on every metric:

```text
                 0        20        40        60        80      100
BLEU        base ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  4.8
            FT   ███████████████████░░░░░░░░░░░░░░░░░░░░░ 46.9   ▲ ~10x
ROUGE-L     base █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 12.1
            FT   █████████████████████████░░░░░░░░░░░░░░░ 61.4   ▲ ~5x
BERTScore   base █████████████████████░░░░░░░░░░░░░░░░░░░ 51.7
            FT   █████████████████████████████████░░░░░░░ 81.4   ▲ +29.7
```

| Metric | 🔴 Base (zero-shot) | 🟢 Fine-tuned |
|:--|:--:|:--:|
| **BLEU** | 4.81 | **46.94** |
| **ROUGE-L** | 12.05 | **61.38** |
| **BERTScore-F1** (tr) | 51.70 | **81.43** |

> **Note:** A large part of the BLEU/ROUGE gain reflects the model learning the dataset's **concise answer format** (the base model is correct but verbose). The **BERTScore** (semantic) gain shows genuine content-similarity improvement. Read the result as *strong alignment to the target instructional style + a semantic-quality gain*.

## 🔧 Model details

| | |
|:--|:--|
| **Base model** | `ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1` (Llama-3, 8B) |
| **Method** | QLoRA (4-bit NF4 + double quant) + NEFTune |
| **LoRA** | `r=16`, `alpha=32`, dropout `0.05`, **all linear layers** (`q/k/v/o/gate/up/down_proj`) |
| **Trainable params** | 41,943,040 / 8,030,261,248 (**0.52%** → **99.48% reduction**) |
| **Effective batch** | 16 · **seq len** 512 (T4) / 1024 (L4·A100) |
| **Optimizer** | `paged_adamw_32bit`, LR `2e-4` cosine, 3 epochs |
| **Hardware** | single GPU (T4 / L4 / A100), auto fp16·bf16 |

## 🚀 Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE    = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"
ADAPTER = "sehinsahfanboy/Turkish-Llama-8B-STEM-QLoRA"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)
tok   = AutoTokenizer.from_pretrained(ADAPTER)

messages = [
    {"role": "system", "content": "Sen bir Türkçe K-12 STEM ve kodlama eğitimi asistanısın. "
                                   "Cevaplarını Türkçe ver, kodda her satırı açıkla."},
    {"role": "user", "content": "Arduino ile servo motor nasıl kontrol edilir?"},
]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
eot = tok.convert_tokens_to_ids("<|eot_id|>")
out = model.generate(ids, max_new_tokens=400, do_sample=True, temperature=0.7,
                     top_p=0.9, eos_token_id=[tok.eos_token_id, eot])
print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
```

## 🎯 Intended use & limitations

- **Intended:** helping students with K–12 STEM/coding questions in Turkish, with short, explained answers.
- **Limitations:** unreliable outside its domain. Trained on a **small (1k), mostly synthetic** dataset, so answers tend to be short and **template-like**, and can be less detailed than the base model on some questions. Code/hardware outputs should be reviewed by a teacher/adult. Inherits biases from the base model.

## 📚 Citation

```bibtex
@misc{stem-tr-2026,
  title  = {STEM TR: Turkish K-12 STEM Instruction Dataset & QLoRA Fine-tuning},
  author = {Alim Kacar},
  year   = {2026},
  note   = {Personal project}
}
```

Methods: **QLoRA** (Dettmers et al., 2023) · **LoRA** (Hu et al., 2021) · **NEFTune** (Jain et al., 2023).
Dataset: [`sehinsahfanboy/stem-tr-instruct-1k`](https://huggingface.co/datasets/sehinsahfanboy/stem-tr-instruct-1k) · Base: `ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1` (Llama-3 license).

<div align="center"><sub>Alim Kacar · 2026</sub></div>
