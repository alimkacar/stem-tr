"""
Değerlendirme Çerçevesi
========================
Fine-tune edilmiş model vs zero-shot baseline karşılaştırması.
Otomatik metrikler (BLEU, ROUGE-L, BERTScore) + LLM-as-Judge.

Kullanım:
    python src/evaluate.py --config configs/config.yaml \
        --adapter outputs/qlora-checkpoints/final-adapter \
        --api-key YOUR_KEY
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import yaml
import torch
import jsonlines
import numpy as np
from tqdm import tqdm

# ─── Config ─────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Model Yükleme ─────────────────────────────────────────

def _dtype():
    """T4 (Turing) bf16 desteklemez -> float16; Ampere+ -> bfloat16 (otomatik)."""
    return torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16


def load_finetuned_model(config: dict, adapter_path: str):
    """Fine-tune edilmiş modeli yükle."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    model_name = config["training"]["base_model"]
    q_config = config["training"]["quantization"]
    compute_dtype = _dtype()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_config["load_in_4bit"],
        bnb_4bit_quant_type=q_config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=q_config["bnb_4bit_use_double_quant"],
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=_dtype(),
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model.eval()

    return model, tokenizer


def load_baseline_model(config: dict):
    """Zero-shot baseline modeli yükle (adapter olmadan)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = config["training"]["base_model"]
    q_config = config["training"]["quantization"]
    compute_dtype = _dtype()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_config["load_in_4bit"],
        bnb_4bit_quant_type=q_config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=q_config["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=_dtype(),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    return model, tokenizer


# ─── Inference ──────────────────────────────────────────────

CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Sen bir Türkçe K-12 STEM ve kodlama eğitimi asistanısın. Öğrencilere Arduino, Scratch, mBlock, robotik, Python ve elektronik konularında yardım ediyorsun. Cevaplarını Türkçe ver, kod içeren cevaplarda her satırı açıkla.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def generate_response(
    model, tokenizer, instruction: str, max_new_tokens: int = 512
) -> str:
    """Tek bir instruction için cevap üret."""
    prompt = CHAT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # Deterministic evaluation
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.15,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def generate_all_responses(
    model, tokenizer, test_data: list[dict], label: str
) -> list[dict]:
    """Test setinin tamamı için cevap üret."""
    results = []
    for ex in tqdm(test_data, desc=f"Inference ({label})"):
        response = generate_response(model, tokenizer, ex["instruction"])
        results.append({
            "id": ex.get("id", ""),
            "instruction": ex["instruction"],
            "reference": ex["output"],
            "prediction": response,
            "category": ex.get("category", ""),
            "difficulty": ex.get("difficulty", ""),
        })
    return results


# ─── Otomatik Metrikler ────────────────────────────────────

def compute_bleu(predictions: list[str], references: list[str]) -> dict:
    """BLEU skoru hesapla."""
    import evaluate
    bleu = evaluate.load("bleu")

    # Tokenize (basit whitespace split)
    results = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )
    return {
        "bleu": round(results["bleu"], 4),
        "bleu_1": round(results["precisions"][0], 4),
        "bleu_2": round(results["precisions"][1], 4),
        "bleu_3": round(results["precisions"][2], 4),
        "bleu_4": round(results["precisions"][3], 4),
    }


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """ROUGE-L skoru hesapla."""
    import evaluate
    rouge = evaluate.load("rouge")

    results = rouge.compute(
        predictions=predictions,
        references=references,
    )
    return {
        "rouge1": round(results["rouge1"], 4),
        "rouge2": round(results["rouge2"], 4),
        "rougeL": round(results["rougeL"], 4),
        "rougeLsum": round(results["rougeLsum"], 4),
    }


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_name: str = "dbmdz/bert-base-turkish-cased",
) -> dict:
    """BERTScore hesapla (Türkçe BERT modeli ile)."""
    from bert_score import score as bert_score

    P, R, F1 = bert_score(
        predictions,
        references,
        model_type=model_name,
        lang="tr",
        verbose=True,
    )
    return {
        "bertscore_precision": round(P.mean().item(), 4),
        "bertscore_recall": round(R.mean().item(), 4),
        "bertscore_f1": round(F1.mean().item(), 4),
    }


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    bertscore_model: str = "dbmdz/bert-base-turkish-cased",
) -> dict:
    """Tüm otomatik metrikleri hesapla."""
    metrics = {}

    print("[EVAL] BLEU hesaplanıyor...")
    metrics.update(compute_bleu(predictions, references))

    print("[EVAL] ROUGE hesaplanıyor...")
    metrics.update(compute_rouge(predictions, references))

    print("[EVAL] BERTScore hesaplanıyor...")
    metrics.update(compute_bertscore(predictions, references, bertscore_model))

    return metrics


# ─── LLM-as-Judge ──────────────────────────────────────────

JUDGE_PROMPT = """Sen bir Türkçe STEM eğitim materyali değerlendirme uzmanısın.

Aşağıdaki soruya verilen cevabı 4 kriter üzerinden 1-5 arası puanla.

SORU:
{instruction}

REFERANS CEVAP:
{reference}

DEĞERLENDİRİLECEK CEVAP:
{prediction}

KRİTERLER (her birini 1-5 arası puanla):
1. faithfulness: Cevap teknik olarak doğru mu? Yanlış bilgi var mı?
2. relevance: Cevap soruyla ne kadar ilgili?
3. completeness: Cevap konuyu yeterince kapsıyor mu?
4. turkish_quality: Türkçe dil kalitesi nasıl? (gramer, akıcılık, terim kullanımı)

JSON formatında cevap ver:
{{"faithfulness": X, "relevance": X, "completeness": X, "turkish_quality": X, "reasoning": "kısa açıklama"}}
"""


def llm_judge_single(
    client,
    instruction: str,
    reference: str,
    prediction: str,
) -> dict:
    """Tek bir örneği Gemini LLM ile değerlendir."""
    prompt = JUDGE_PROMPT.format(
        instruction=instruction,
        reference=reference[:800],
        prediction=prediction[:800],
    )

    try:
        response = client.generate_content(
            prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 300},
        )
        content = response.text.strip()

        # JSON parse
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        scores = json.loads(content.strip())
        return scores

    except Exception as e:
        print(f"[WARN] LLM judge hatası: {e}")
        return {
            "faithfulness": None,
            "relevance": None,
            "completeness": None,
            "turkish_quality": None,
            "reasoning": f"Error: {str(e)}",
        }


def run_llm_judge(
    results: list[dict],
    api_key: str,
    config: dict,
    max_samples: int = 50,
) -> list[dict]:
    """LLM-as-Judge değerlendirmesi (Gemini)."""
    import google.generativeai as genai
    import time

    genai.configure(api_key=api_key)
    judge_model = config["evaluation"]["llm_judge"].get("model", "gemini-2.0-flash")
    client = genai.GenerativeModel(judge_model)

    # Örnek sayısını sınırla (maliyet kontrolü)
    samples = results[:max_samples]

    print(f"\n[EVAL] LLM-as-Judge başlıyor ({len(samples)} örnek, model: {judge_model})")

    judged = []
    for ex in tqdm(samples, desc="LLM Judge"):
        scores = llm_judge_single(
            client=client,
            instruction=ex["instruction"],
            reference=ex["reference"],
            prediction=ex["prediction"],
        )
        ex["judge_scores"] = scores
        judged.append(ex)
        time.sleep(0.5)  # Rate limiting

    # Ortalama skorları hesapla
    criteria = ["faithfulness", "relevance", "completeness", "turkish_quality"]
    avg_scores = {}
    for c in criteria:
        values = [
            j["judge_scores"][c]
            for j in judged
            if j["judge_scores"].get(c) is not None
        ]
        if values:
            avg_scores[f"judge_{c}_mean"] = round(np.mean(values), 2)
            avg_scores[f"judge_{c}_std"] = round(np.std(values), 2)

    return judged, avg_scores


# ─── Karşılaştırma Raporu ──────────────────────────────────

def generate_comparison_report(
    ft_metrics: dict,
    baseline_metrics: dict,
    ft_judge: dict,
    baseline_judge: dict,
    output_path: str,
):
    """Fine-tuned vs baseline karşılaştırma raporu."""
    report = {
        "model_comparison": {
            "finetuned": {
                "automatic_metrics": ft_metrics,
                "llm_judge_scores": ft_judge,
            },
            "baseline_zero_shot": {
                "automatic_metrics": baseline_metrics,
                "llm_judge_scores": baseline_judge,
            },
        },
        "improvements": {},
    }

    # Delta hesapla
    for key in ft_metrics:
        if key in baseline_metrics:
            delta = ft_metrics[key] - baseline_metrics[key]
            pct = (delta / baseline_metrics[key] * 100) if baseline_metrics[key] != 0 else 0
            report["improvements"][key] = {
                "delta": round(delta, 4),
                "improvement_pct": round(pct, 2),
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Güzel yazdır
    print("\n" + "=" * 70)
    print("DEĞERLENDİRME SONUÇLARI")
    print("=" * 70)
    print(f"\n{'Metrik':<25} {'Fine-tuned':>12} {'Baseline':>12} {'Delta':>10}")
    print("-" * 60)

    for key in ft_metrics:
        ft_val = ft_metrics[key]
        bl_val = baseline_metrics.get(key, "N/A")
        delta = ft_val - bl_val if isinstance(bl_val, (int, float)) else "N/A"
        delta_str = f"{delta:+.4f}" if isinstance(delta, float) else delta
        print(f"{key:<25} {ft_val:>12.4f} {bl_val:>12.4f} {delta_str:>10}")

    print("=" * 70)
    print(f"\nRapor kaydedildi: {output_path}")

    return report


# ─── Kategori Bazlı Analiz ─────────────────────────────────

def category_analysis(results: list[dict], output_dir: str):
    """Kategori ve zorluk bazlı performans analizi."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'

        # Kategori bazlı judge skorları
        categories = {}
        for r in results:
            cat = r.get("category", "other")
            scores = r.get("judge_scores", {})
            if scores.get("faithfulness") is not None:
                categories.setdefault(cat, []).append(scores)

        if not categories:
            print("[WARN] Kategori analizi için yeterli veri yok.")
            return

        # Grafik
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Kategori bazlı ortalama skorlar
        cat_names = list(categories.keys())
        criteria = ["faithfulness", "relevance", "completeness", "turkish_quality"]
        x = np.arange(len(cat_names))
        width = 0.2

        for i, criterion in enumerate(criteria):
            means = []
            for cat in cat_names:
                vals = [s[criterion] for s in categories[cat] if s.get(criterion)]
                means.append(np.mean(vals) if vals else 0)
            axes[0].bar(x + i * width, means, width, label=criterion)

        axes[0].set_xlabel("Kategori")
        axes[0].set_ylabel("Ortalama Skor (1-5)")
        axes[0].set_title("Kategori Bazlı LLM-Judge Skorları")
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(cat_names, rotation=45, ha="right")
        axes[0].legend(fontsize=8)
        axes[0].set_ylim(0, 5.5)

        # 2. Zorluk bazlı
        difficulties = {}
        for r in results:
            diff = r.get("difficulty", "other")
            scores = r.get("judge_scores", {})
            if scores.get("faithfulness") is not None:
                difficulties.setdefault(diff, []).append(scores)

        if difficulties:
            diff_names = list(difficulties.keys())
            x2 = np.arange(len(diff_names))
            for i, criterion in enumerate(criteria):
                means = []
                for diff in diff_names:
                    vals = [s[criterion] for s in difficulties[diff] if s.get(criterion)]
                    means.append(np.mean(vals) if vals else 0)
                axes[1].bar(x2 + i * width, means, width, label=criterion)

            axes[1].set_xlabel("Zorluk Seviyesi")
            axes[1].set_ylabel("Ortalama Skor (1-5)")
            axes[1].set_title("Zorluk Bazlı LLM-Judge Skorları")
            axes[1].set_xticks(x2 + width * 1.5)
            axes[1].set_xticklabels(diff_names)
            axes[1].legend(fontsize=8)
            axes[1].set_ylim(0, 5.5)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "category_analysis.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[INFO] Kategori analizi: {plot_path}")

    except ImportError:
        print("[WARN] matplotlib gerekli, grafik oluşturulamadı.")


# ─── Ana Pipeline ──────────────────────────────────────────

def run_evaluation(
    config_path: str,
    adapter_path: str,
    api_key: Optional[str] = None,
    test_path: str = "data/splits/test.jsonl",
    output_dir: str = "outputs/evaluation",
    skip_baseline: bool = False,
    skip_judge: bool = False,
):
    """Tam değerlendirme pipeline'ı."""
    config = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)

    # Test verisini yükle
    with jsonlines.open(test_path) as reader:
        test_data = list(reader)
    print(f"[INFO] Test seti: {len(test_data)} örnek")

    # ── Fine-tuned model ──
    print("\n[1/4] Fine-tuned model yükleniyor...")
    ft_model, ft_tokenizer = load_finetuned_model(config, adapter_path)
    ft_results = generate_all_responses(ft_model, ft_tokenizer, test_data, "fine-tuned")

    # Belleği boşalt
    del ft_model
    torch.cuda.empty_cache()

    # ── Baseline model ──
    if not skip_baseline:
        print("\n[2/4] Baseline model yükleniyor...")
        bl_model, bl_tokenizer = load_baseline_model(config)
        bl_results = generate_all_responses(bl_model, bl_tokenizer, test_data, "baseline")
        del bl_model
        torch.cuda.empty_cache()
    else:
        bl_results = None

    # ── Otomatik metrikler ──
    print("\n[3/4] Otomatik metrikler hesaplanıyor...")
    ft_preds = [r["prediction"] for r in ft_results]
    ft_refs = [r["reference"] for r in ft_results]
    ft_metrics = compute_all_metrics(
        ft_preds, ft_refs, config["evaluation"]["bertscore_model"]
    )

    if bl_results:
        bl_preds = [r["prediction"] for r in bl_results]
        bl_refs = [r["reference"] for r in bl_results]
        bl_metrics = compute_all_metrics(
            bl_preds, bl_refs, config["evaluation"]["bertscore_model"]
        )
    else:
        bl_metrics = {}

    # ── LLM-as-Judge ──
    ft_judge_scores, bl_judge_scores = {}, {}
    if not skip_judge and api_key:
        print("\n[4/4] LLM-as-Judge değerlendirmesi...")
        ft_results, ft_judge_scores = run_llm_judge(
            ft_results, api_key, config
        )
        if bl_results:
            bl_results, bl_judge_scores = run_llm_judge(
                bl_results, api_key, config
            )
    else:
        print("\n[4/4] LLM-as-Judge atlandı (--api-key verilmedi veya --skip-judge)")

    # ── Rapor ──
    report = generate_comparison_report(
        ft_metrics, bl_metrics,
        ft_judge_scores, bl_judge_scores,
        os.path.join(output_dir, "evaluation_report.json"),
    )

    # Detaylı sonuçları kaydet
    with jsonlines.open(os.path.join(output_dir, "ft_predictions.jsonl"), "w") as w:
        w.write_all(ft_results)
    if bl_results:
        with jsonlines.open(os.path.join(output_dir, "bl_predictions.jsonl"), "w") as w:
            w.write_all(bl_results)

    # Kategori analizi
    if ft_judge_scores:
        category_analysis(ft_results, output_dir)

    print(f"\n[OK] Değerlendirme tamamlandı. Sonuçlar: {output_dir}/")
    return report


# ─── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Değerlendirme")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--adapter", required=True, help="Fine-tuned adapter yolu")
    parser.add_argument("--test", default="data/splits/test.jsonl")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--api-key", default=None, help="LLM Judge için API key")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")

    run_evaluation(
        config_path=args.config,
        adapter_path=args.adapter,
        api_key=api_key,
        test_path=args.test,
        output_dir=args.output_dir,
        skip_baseline=args.skip_baseline,
        skip_judge=args.skip_judge,
    )
