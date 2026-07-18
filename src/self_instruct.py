"""
Self-Instruct Pipeline - Türkçe STEM/Kodlama Veri Seti Genişletme
================================================================
Seed örneklerden yola çıkarak LLM ile yeni instruction-output çiftleri üretir.
Wang et al. (2023) Self-Instruct yaklaşımının Türkçe STEM adaptasyonu.

Kullanım:
    python src/self_instruct.py --config configs/config.yaml --api-key YOUR_KEY
"""

import os
import json
import random
import argparse
import time
from pathlib import Path
from typing import Optional

import yaml
import jsonlines
from tqdm import tqdm

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ─── Sabitler ───────────────────────────────────────────────
SYSTEM_PROMPT = """Sen bir Türkçe K-12 STEM ve kodlama eğitimi uzmanısın.
Görevin, verilen örneklere benzer ama farklı yeni eğitim materyalleri üretmek.

Kurallar:
1. Türkçe yaz, teknik terimleri doğru kullan.
2. Hedef kitle: ilkokul (1-4), ortaokul (5-8) veya lise (9-12) öğrencileri.
3. Kategoriler: Arduino, mBlock, Scratch, robotik, Python STEM, elektronik, algoritma.
4. Her örnek bir "instruction" (soru/görev) ve "output" (detaylı cevap) içermeli.
5. Kod içeren cevaplarda her satırı Türkçe açıkla.
6. Örnekleri birbirinden farklı tut — aynı soruyu tekrarlama.
7. Zorluk seviyesini belirt.
"""

GENERATION_PROMPT_TEMPLATE = """Aşağıda {category} kategorisinde Türkçe STEM/kodlama eğitimi örnekleri var.
Bu örneklerden ilham alarak {count} TANE YENİ ve FARKLI örnek üret.

Zorluk seviyesi: {difficulty}
Kategori: {category}

─── ÖRNEK REFERANSLAR ───
{seed_examples}
─── ÖRNEK REFERANSLAR SONU ───

Yukarıdaki örneklerle AYNI soruları sorma. Farklı konular ve senaryolar üret.

Her örneği şu JSON formatında yaz (her satıra bir JSON):
{{"instruction": "...", "input": "", "output": "..."}}

Sadece JSON satırlarını yaz, başka açıklama ekleme.
"""


def load_config(config_path: str) -> dict:
    """YAML config dosyasını yükle."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_seeds(seed_path: str) -> list[dict]:
    """Seed örneklerini JSONL dosyasından yükle."""
    seeds = []
    with jsonlines.open(seed_path, mode="r") as reader:
        for item in reader:
            seeds.append(item)
    print(f"[INFO] {len(seeds)} seed örnek yüklendi: {seed_path}")
    return seeds


def format_seed_examples(seeds: list[dict], num_examples: int = 3) -> str:
    """Rastgele seed örneklerini prompt'a eklemek için formatla."""
    selected = random.sample(seeds, min(num_examples, len(seeds)))
    formatted = []
    for i, s in enumerate(selected, 1):
        formatted.append(
            f"Örnek {i}:\n"
            f"  Instruction: {s['instruction']}\n"
            f"  Output: {s['output'][:500]}..."  # Çok uzun çıktıları kırp
        )
    return "\n\n".join(formatted)


def generate_batch(
    client,
    seeds: list[dict],
    category: str,
    difficulty: str,
    config: dict,
    count: int = 5,
) -> list[dict]:
    """Bir batch yeni örnek üret (Gemini API)."""
    si_config = config["self_instruct"]

    # Aynı kategoriden seed'leri tercih et, yoksa rastgele seç
    category_seeds = [s for s in seeds if s.get("category") == category]
    if len(category_seeds) < si_config["num_seeds_per_prompt"]:
        category_seeds = seeds

    seed_text = format_seed_examples(category_seeds, si_config["num_seeds_per_prompt"])

    prompt = GENERATION_PROMPT_TEMPLATE.format(
        category=category,
        difficulty=difficulty,
        count=count,
        seed_examples=seed_text,
    )

    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt

    for attempt in range(si_config["max_retries"]):
        try:
            response = client.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=si_config["temperature"],
                    max_output_tokens=si_config["max_tokens"],
                ),
            )

            content = response.text.strip()
            return parse_generated_examples(content, category, difficulty)

        except Exception as e:
            wait_time = 2 ** (attempt + 1)
            print(f"[WARN] API hatası (deneme {attempt+1}): {e}. {wait_time}s bekleniyor...")
            time.sleep(wait_time)

    print(f"[ERROR] {si_config['max_retries']} denemede başarısız oldu.")
    return []


def parse_generated_examples(
    raw_text: str, category: str, difficulty: str
) -> list[dict]:
    """LLM çıktısını parse et, geçerli JSON satırlarını döndür."""
    examples = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if "instruction" in obj and "output" in obj:
                obj["category"] = category
                obj["difficulty"] = difficulty
                obj.setdefault("input", "")
                examples.append(obj)
        except json.JSONDecodeError:
            continue
    return examples


# ─── Deduplikasyon ──────────────────────────────────────────
def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Basit ROUGE-L (LCS tabanlı) benzerlik skoru."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def deduplicate(
    new_examples: list[dict],
    existing_examples: list[dict],
    threshold: float = 0.85,
) -> list[dict]:
    """Mevcut örneklere çok benzer olanları filtrele."""
    existing_instructions = [e["instruction"] for e in existing_examples]
    unique = []

    for ex in new_examples:
        is_duplicate = False
        for existing_instr in existing_instructions:
            similarity = compute_rouge_l(existing_instr, ex["instruction"])
            if similarity >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(ex)
            existing_instructions.append(ex["instruction"])

    removed = len(new_examples) - len(unique)
    if removed > 0:
        print(f"[DEDUP] {removed} duplike örnek filtrelendi.")
    return unique


# ─── Ana Pipeline ───────────────────────────────────────────
def run_pipeline(
    config_path: str,
    api_key: str,
    output_path: Optional[str] = None,
    seed_path: Optional[str] = None,
):
    """Self-Instruct pipeline'ını çalıştır."""
    config = load_config(config_path)
    data_config = config["data"]
    si_config = config["self_instruct"]

    # Yolları ayarla
    if seed_path is None:
        seed_path = "data/seeds/seed_examples.jsonl"
    if output_path is None:
        output_path = "data/generated_raw.jsonl"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Gemini client
    if not HAS_GEMINI:
        raise ImportError("google-generativeai paketi gerekli: pip install google-generativeai")
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(si_config.get("model", "gemini-2.0-flash"))

    # Seed'leri yükle
    seeds = load_seeds(seed_path)
    all_generated = list(seeds)  # Seed'lerle başla (deduplicate için)

    # Hedef: target_total - seed_count kadar yeni örnek
    target_new = data_config["target_total"] - len(seeds)
    # Rejection rate'i hesaba kat → daha fazla üret
    target_with_buffer = int(target_new / (1 - data_config["rejection_rate"]))

    print(f"\n{'='*60}")
    print(f"Self-Instruct Pipeline Başlatılıyor")
    print(f"{'='*60}")
    print(f"  Seed sayısı:        {len(seeds)}")
    print(f"  Hedef yeni örnek:   {target_new}")
    print(f"  Buffer ile üretim:  {target_with_buffer}")
    print(f"  Kategoriler:        {data_config['categories']}")
    print(f"  LLM modeli:         {si_config['model']}")
    print(f"{'='*60}\n")

    generated_count = 0
    batch_size = si_config["batch_size"]
    categories = data_config["categories"]
    difficulties = data_config["difficulty_levels"]

    pbar = tqdm(total=target_with_buffer, desc="Üretim", unit="örnek")

    while generated_count < target_with_buffer:
        # Rastgele kategori ve zorluk seç
        category = random.choice(categories)
        difficulty = random.choice(difficulties)

        # Batch üret
        batch = generate_batch(
            client=client,
            seeds=seeds,
            category=category,
            difficulty=difficulty,
            config=config,
            count=batch_size,
        )

        if batch:
            # Deduplicate
            unique_batch = deduplicate(
                batch, all_generated, si_config["similarity_threshold"]
            )
            all_generated.extend(unique_batch)
            generated_count += len(unique_batch)
            pbar.update(len(unique_batch))

        # Rate limiting
        time.sleep(1)

    pbar.close()

    # Sonuçları kaydet (seed'leri hariç tut)
    new_examples = all_generated[len(seeds):]

    # ID ata
    for i, ex in enumerate(new_examples):
        ex["id"] = f"gen_{i+1:04d}"
        ex["source"] = "self_instruct"

    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(new_examples)

    print(f"\n[OK] {len(new_examples)} yeni örnek kaydedildi: {output_path}")
    print(f"[INFO] Sonraki adım: Human-in-the-loop filtreleme (src/filter_hitl.py)")

    return new_examples


# ─── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-Instruct ile Türkçe STEM veri seti genişletme"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config dosyası yolu"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API anahtarı (veya GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--output", type=str, default="data/generated_raw.jsonl", help="Çıktı dosyası"
    )
    parser.add_argument(
        "--seeds", type=str, default="data/seeds/seed_examples.jsonl", help="Seed dosyası"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API key gerekli: --api-key veya GEMINI_API_KEY env var")

    run_pipeline(
        config_path=args.config,
        api_key=api_key,
        output_path=args.output,
        seed_path=args.seeds,
    )
