"""
QLoRA Fine-tuning - Turkish-Llama-8b-Instruct
=============================================
4-bit quantization (QLoRA) ile bellek-verimli fine-tuning.

Sürüm dayanıklılığı için TRL KULLANMAZ; doğrudan `transformers.Trainer`
kullanır ve yanıt-yalnızca (completion-only) maskeleme elle yapılır.
Bu, Colab'ın güncel transformers/torch sürümleriyle uyumu korur.

bitsandbytes yüklenemezse otomatik olarak bf16/fp16 LoRA'ya düşer
(eğitim yine de çalışır; 4-bit yoksa sadece daha çok bellek kullanır).

Kullanım (script):
    python src/qlora_finetune.py --config configs/config.yaml --data-dir data/splits

Kullanım (notebook):
    from src.qlora_finetune import load_config, setup_and_train
    cfg = load_config("configs/config.yaml")
    trainer = setup_and_train(cfg, data_dir="data/splits")
"""

import os
import json
import time
import argparse
import inspect

import yaml
import torch
import jsonlines
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


# ─── Yardımcılar ────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _bf16_ok() -> bool:
    """bf16 yalnızca Ampere+ (compute capability >= 8.0) mimaride native çalışır.
    T4 = 7.5 -> fp16 (daha hızlı ve daha az bellek)."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability()[0] >= 8
    except Exception:
        return torch.cuda.is_bf16_supported()


def _dtype():
    return torch.bfloat16 if _bf16_ok() else torch.float16


# ─── Chat template & formatlama ─────────────────────────────

CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Sen bir Türkçe K-12 STEM ve kodlama eğitimi asistanısın. Öğrencilere Arduino, Scratch, mBlock, robotik, Python ve elektronik konularında yardım ediyorsun. Cevaplarını Türkçe ver, kod içeren cevaplarda her satırı açıkla.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}{input_section}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

# Yanıtın başladığı işaret (bu noktaya kadar olan tokenlar maskelenir)
RESPONSE_MARKER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def format_example(example: dict) -> str:
    """Bir örneği tam chat metnine çevir."""
    input_section = ""
    if example.get("input", "").strip():
        input_section = f"\n\nEk bilgi: {example['input']}"
    return CHAT_TEMPLATE.format(
        instruction=example["instruction"],
        input_section=input_section,
        output=example["output"],
    )


def load_split(data_dir: str, key: str, fname: str):
    path = os.path.join(data_dir, f"{fname}.jsonl")
    if not os.path.exists(path):
        print(f"[WARN] {path} bulunamadı, atlanıyor.")
        return None
    with jsonlines.open(path) as reader:
        rows = list(reader)
    print(f"[INFO] {key} ({fname}.jsonl): {len(rows)} örnek yüklendi")
    return rows


# ─── Tokenizasyon + yanıt-yalnızca maskeleme ────────────────

def build_tokenized_dataset(rows, tokenizer, max_len: int) -> Dataset:
    """Her örneği tokenize eder; sadece asistan cevabı üzerinde loss (labels)
    hesaplanacak şekilde soru/sistem kısmını -100 ile maskeler."""

    def encode(ex):
        full_text = format_example(ex)
        cut = full_text.find(RESPONSE_MARKER)
        prompt_text = full_text[: cut + len(RESPONSE_MARKER)]

        full_ids = tokenizer(
            full_text, add_special_tokens=False,
            truncation=True, max_length=max_len,
        )["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        n_mask = min(len(prompt_ids), len(full_ids))
        labels = [-100] * n_mask + full_ids[n_mask:]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    encoded = [encode(r) for r in rows]
    # Tamamı maskeli (cevabı truncate olmuş) örnekleri at
    encoded = [e for e in encoded if any(l != -100 for l in e["labels"])]
    return Dataset.from_list(encoded)


# ─── Model & tokenizer ──────────────────────────────────────

def setup_quantization(config: dict) -> BitsAndBytesConfig:
    q = config["training"]["quantization"]
    print(f"[INFO] Compute dtype: {'bfloat16' if _bf16_ok() else 'float16 (T4/Turing)'}")
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=_dtype(),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )


def _is_transient(err) -> bool:
    """HuggingFace/ağ kaynaklı geçici hata mı (bitsandbytes ile ilgisiz)?"""
    s = str(err).lower()
    return any(k in s for k in [
        "429", "too many requests", "rate limit", "timeout", "timed out",
        "connection", "temporarily", "max retries", "queue size", "503", "502",
    ])


def _load_4bit(model_name, config, retries=4):
    """4-bit modeli yükler; geçici (429/ağ) hatalarda yeniden dener."""
    bnb_config = setup_quantization(config)
    last = None
    for attempt in range(retries):
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config,
                device_map="auto", trust_remote_code=True,
            )
            print("[INFO] 4-bit (QLoRA) yükleme başarılı.")
            return m
        except Exception as e:
            last = e
            if _is_transient(e) and attempt < retries - 1:
                wait = 30 * (attempt + 1)
                print(f"[WARN] Geçici hata (deneme {attempt+1}/{retries}); {wait}s bekleniyor...")
                print(f"       {type(e).__name__}: {str(e)[:160]}")
                time.sleep(wait)
                continue
            raise
    raise last


def load_model_and_tokenizer(config: dict):
    """Modeli 4-bit (QLoRA) yüklemeyi dener; geçici HF hatalarında yeniden dener.
    Yalnızca bitsandbytes GERÇEKTEN kullanılamıyorsa bf16/fp16 LoRA'ya düşer.
    Döner: (model, tokenizer, quantized)."""
    model_name = config["training"]["base_model"]
    print(f"\n[INFO] Model yükleniyor: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # bitsandbytes gerçekten kurulu/çalışır mı?
    try:
        import bitsandbytes  # noqa: F401
        bnb_ok = True
    except Exception as e:
        bnb_ok = False
        print(f"[WARN] bitsandbytes içe aktarılamadı: {type(e).__name__}: {e}")

    quantized = False
    model = None
    if bnb_ok:
        try:
            model = _load_4bit(model_name, config)
            quantized = True
        except Exception as e:
            if _is_transient(e):
                # Ağ hatasını sessizce bf16'ya düşürme (16GB offload'a yol açar).
                raise RuntimeError(
                    "HuggingFace geçici hata (ör. 429 rate-limit) nedeniyle model "
                    "indirilemedi. Model dosyaları büyük olasılıkla önbelleğe alındı; "
                    "lütfen bu hücreyi TEKRAR çalıştırın (2. denemede 4-bit yüklenir)."
                ) from e
            print(f"\n[WARN] 4-bit yüklenemedi (bitsandbytes/CUDA): {type(e).__name__}: {str(e)[:200]}")

    if model is None:
        print("[WARN] bf16/fp16 (kuantizasyonsuz) LoRA'ya geçiliyor. "
              "NOT: 8B model ~16GB'dir; küçük GPU'da CPU'ya taşabilir (yavaş).")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=_dtype(),
            device_map="auto", trust_remote_code=True,
        )

    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    if quantized:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Toplam parametre: {total:,}  | Kuantizasyon: {'4-bit NF4' if quantized else 'yok (bf16/fp16)'}")
    return model, tokenizer, quantized


def setup_lora(model, config: dict):
    lc = config["training"]["lora"]
    peft_config = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        target_modules=lc["target_modules"],
        bias=lc["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config


# ─── Training arguments (sürüm-dayanıklı) ───────────────────

def setup_training_args(config: dict, quantized: bool = True) -> TrainingArguments:
    args = config["training"]["args"]
    output_dir = config["training"]["output_dir"]

    # paged_adamw_32bit bitsandbytes gerektirir; kuantizasyon yoksa standart optim.
    optim = args["optim"] if quantized else "adamw_torch"

    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=args["num_train_epochs"],
        per_device_train_batch_size=args["per_device_train_batch_size"],
        per_device_eval_batch_size=args["per_device_train_batch_size"],
        gradient_accumulation_steps=args["gradient_accumulation_steps"],
        learning_rate=args["learning_rate"],
        weight_decay=args["weight_decay"],
        warmup_ratio=args["warmup_ratio"],
        lr_scheduler_type=args["lr_scheduler_type"],
        logging_steps=args["logging_steps"],
        save_steps=args["save_steps"],
        save_strategy="steps",
        fp16=not _bf16_ok(),
        bf16=_bf16_ok(),
        gradient_checkpointing=args["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optim,
        seed=args["seed"],
        report_to="none",
        save_total_limit=3,
    )

    fields = getattr(TrainingArguments, "__dataclass_fields__", {})
    # transformers sürümüne göre eval argümanının adı değişti
    eval_key = "eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"
    kwargs[eval_key] = "steps"
    kwargs["eval_steps"] = args["eval_steps"]
    # en iyi modeli sonda yükle (destekleniyorsa)
    if "load_best_model_at_end" in fields:
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False
    # NEFTune (kalite artışı) — destekleniyorsa ve config'te tanımlıysa
    neft = args.get("neftune_noise_alpha", 0)
    if neft and "neftune_noise_alpha" in fields:
        kwargs["neftune_noise_alpha"] = neft

    # Bilinmeyen bir argüman olursa güvenli şekilde çıkar
    valid = {k: v for k, v in kwargs.items() if (not fields) or k in fields}
    return TrainingArguments(**valid)


# ─── Ana eğitim ─────────────────────────────────────────────

def setup_and_train(config: dict, data_dir: str = "data/splits"):
    print("=" * 60)
    print("QLoRA Fine-tuning Pipeline (transformers.Trainer)")
    print("=" * 60)

    train_rows = load_split(data_dir, "train", "train")
    val_rows = load_split(data_dir, "val", "validation")
    if not train_rows:
        raise ValueError(f"Train seti bulunamadı: {data_dir}/train.jsonl")

    model, tokenizer, quantized = load_model_and_tokenizer(config)
    model, _ = setup_lora(model, config)

    max_len = config["training"]["args"]["max_seq_length"]
    train_ds = build_tokenized_dataset(train_rows, tokenizer, max_len)
    eval_ds = build_tokenized_dataset(val_rows, tokenizer, max_len) if val_rows else None
    print(f"[INFO] Tokenize edildi -> train: {len(train_ds)}"
          + (f", val: {len(eval_ds)}" if eval_ds is not None else ""))

    training_args = setup_training_args(config, quantized=quantized)
    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    print("\n[INFO] Eğitim başlıyor...")
    train_result = trainer.train()

    output_dir = config["training"]["output_dir"]
    final_dir = os.path.join(output_dir, "final-adapter")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    print(f"\n[OK] Adapter kaydedildi: {final_dir}")
    plot_training_curve(trainer, output_dir)
    return trainer


# ─── Görselleştirme ─────────────────────────────────────────

def plot_training_curve(trainer, output_dir: str):
    try:
        import matplotlib.pyplot as plt
        logs = trainer.state.log_history
        tr = [(l["step"], l["loss"]) for l in logs if "loss" in l]
        ev = [(l["step"], l["eval_loss"]) for l in logs if "eval_loss" in l]
        if not tr:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        s, y = zip(*tr); ax.plot(s, y, label="Train Loss", alpha=0.7)
        if ev:
            es, ey = zip(*ev); ax.plot(es, ey, label="Eval Loss", marker="o", markersize=4)
        ax.set_xlabel("Adım"); ax.set_ylabel("Loss")
        ax.set_title("QLoRA Fine-tuning - Training Eğrisi")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_curve.png"), dpi=150)
        plt.close()
        print(f"[INFO] Training eğrisi: {output_dir}/training_curve.png")
    except Exception as e:
        print(f"[WARN] Grafik oluşturulamadı: {e}")


# ─── Inference testi ────────────────────────────────────────

def test_inference(adapter_path: str, config: dict, test_prompts=None):
    from peft import PeftModel
    if test_prompts is None:
        test_prompts = [
            "Arduino ile servo motor nasıl kontrol edilir?",
            "Scratch'te bir labirent oyunu nasıl yapılır?",
            "Python'da bir listeyi bubble sort ile sırala ve açıkla.",
        ]
    model_name = config["training"]["base_model"]
    try:
        base = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=setup_quantization(config), device_map="auto"
        )
    except Exception:
        base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=_dtype(), device_map="auto"
        )
    model = PeftModel.from_pretrained(base, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model.eval()

    print("\n" + "=" * 60 + "\nInference Test\n" + "=" * 60)
    for prompt in test_prompts:
        text = CHAT_TEMPLATE.format(instruction=prompt, input_section="", output="")
        text = text.split(RESPONSE_MARKER)[0] + RESPONSE_MARKER
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=512, temperature=0.7, top_p=0.9,
                do_sample=True, repetition_penalty=1.15,
                eos_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n[SORU] {prompt}\n[CEVAP] {resp[:500]}...\n" + "-" * 40)
    return model, tokenizer


# ─── CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning (TRL-free)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-dir", default="data/splits")
    parser.add_argument("--test-only", type=str, default=None,
                        help="Sadece inference test (adapter yolu)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.test_only:
        test_inference(args.test_only, config)
    else:
        setup_and_train(config, data_dir=args.data_dir)
