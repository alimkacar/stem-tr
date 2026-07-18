"""
Human-in-the-Loop Filtreleme Aracı
===================================
Self-Instruct ile üretilen örnekleri kalite kontrolünden geçirir.
Otomatik filtreler + interaktif inceleme arayüzü.

Kullanım:
    python src/filter_hitl.py --input data/generated_raw.jsonl --output data/filtered.jsonl
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import yaml
import jsonlines
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import print as rprint

console = Console()


def load_config(config_path: str) -> dict:
    """YAML config dosyasını yükle."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Otomatik Kalite Filtreleri ─────────────────────────────

def check_length(example: dict, min_instruction: int = 15, min_output: int = 50) -> tuple[bool, str]:
    """Minimum uzunluk kontrolü."""
    instr_len = len(example.get("instruction", ""))
    out_len = len(example.get("output", ""))
    if instr_len < min_instruction:
        return False, f"Instruction çok kısa ({instr_len} karakter, min: {min_instruction})"
    if out_len < min_output:
        return False, f"Output çok kısa ({out_len} karakter, min: {min_output})"
    return True, "OK"


def check_language(example: dict) -> tuple[bool, str]:
    """Temel Türkçe dil kontrolü - Türkçe karakterler içeriyor mu?"""
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    text = example.get("instruction", "") + example.get("output", "")
    has_turkish = any(c in turkish_chars for c in text)
    if not has_turkish:
        return False, "Türkçe karakter bulunamadı"
    return True, "OK"


def check_structure(example: dict) -> tuple[bool, str]:
    """Yapısal bütünlük kontrolü."""
    required_keys = ["instruction", "output"]
    for key in required_keys:
        if key not in example or not example[key].strip():
            return False, f"Eksik veya boş alan: {key}"
    return True, "OK"


def check_code_blocks(example: dict) -> tuple[bool, str]:
    """Kod içeren çıktılarda code block formatı kontrolü."""
    output = example.get("output", "")
    # Kod anahtar kelimeleri varsa ama code block yoksa uyar
    code_keywords = ["def ", "void ", "int ", "print(", "digitalWrite", "import "]
    has_code = any(kw in output for kw in code_keywords)
    has_block = "```" in output
    if has_code and not has_block:
        return False, "Kod var ama code block (```) formatı kullanılmamış"
    return True, "OK"


def check_repetition(example: dict, threshold: float = 0.3) -> tuple[bool, str]:
    """Aşırı tekrar kontrolü - aynı cümlenin tekrarlanması."""
    output = example.get("output", "")
    sentences = [s.strip() for s in output.split(".") if len(s.strip()) > 20]
    if len(sentences) < 3:
        return True, "OK"
    unique = set(sentences)
    repetition_rate = 1 - (len(unique) / len(sentences))
    if repetition_rate > threshold:
        return False, f"Yüksek tekrar oranı: {repetition_rate:.1%}"
    return True, "OK"


AUTO_FILTERS = [
    ("Uzunluk", check_length),
    ("Dil", check_language),
    ("Yapı", check_structure),
    ("Kod Formatı", check_code_blocks),
    ("Tekrar", check_repetition),
]


def run_auto_filters(example: dict) -> tuple[bool, list[str]]:
    """Tüm otomatik filtreleri çalıştır."""
    issues = []
    for name, filter_fn in AUTO_FILTERS:
        passed, msg = filter_fn(example)
        if not passed:
            issues.append(f"[{name}] {msg}")
    return len(issues) == 0, issues


# ─── İnteraktif İnceleme ───────────────────────────────────

def display_example(example: dict, index: int, total: int):
    """Bir örneği güzel formatlı göster."""
    console.clear()

    # Üst bilgi
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("ID", example.get("id", "N/A"))
    table.add_row("Kategori", example.get("category", "N/A"))
    table.add_row("Zorluk", example.get("difficulty", "N/A"))
    table.add_row("Kaynak", example.get("source", "N/A"))

    console.print(f"\n[bold]─── Örnek {index+1}/{total} ───[/bold]")
    console.print(table)

    # Instruction
    console.print(Panel(
        example.get("instruction", ""),
        title="[bold green]INSTRUCTION[/bold green]",
        border_style="green",
    ))

    # Output (kırpılmış)
    output_text = example.get("output", "")
    if len(output_text) > 1000:
        output_text = output_text[:1000] + "\n\n... [kırpıldı, tam metin dosyada]"

    console.print(Panel(
        output_text,
        title="[bold blue]OUTPUT[/bold blue]",
        border_style="blue",
    ))


def interactive_review(
    examples: list[dict],
    auto_passed: list[dict],
    auto_failed: list[dict],
) -> tuple[list[dict], list[dict]]:
    """İnteraktif inceleme arayüzü."""
    console.print(f"\n[bold yellow]═══ İnteraktif İnceleme ═══[/bold yellow]")
    console.print(f"Otomatik filtreyi geçen: {len(auto_passed)}")
    console.print(f"Otomatik reddedilen:     {len(auto_failed)}")
    console.print(f"\nŞimdi otomatik filtreyi geçen örnekleri inceleyeceksiniz.\n")

    accepted = []
    rejected = []
    skip_rest = False

    for i, example in enumerate(auto_passed):
        if skip_rest:
            accepted.append(example)
            continue

        display_example(example, i, len(auto_passed))

        choice = Prompt.ask(
            "\n[bold]Karar[/bold]",
            choices=["k", "r", "d", "t", "q"],
            default="k",
        )
        # k=kabul, r=red, d=düzenle, t=tümünü kabul, q=çık

        if choice == "k":
            example["hitl_status"] = "accepted"
            accepted.append(example)
        elif choice == "r":
            reason = Prompt.ask("Red sebebi (opsiyonel)", default="kalite yetersiz")
            example["hitl_status"] = "rejected"
            example["rejection_reason"] = reason
            rejected.append(example)
        elif choice == "d":
            console.print("[yellow]Düzenleme modu (boş bırakırsan değişmez):[/yellow]")
            new_instr = Prompt.ask("Yeni instruction", default="")
            new_output = Prompt.ask("Yeni output", default="")
            if new_instr:
                example["instruction"] = new_instr
            if new_output:
                example["output"] = new_output
            example["hitl_status"] = "edited"
            accepted.append(example)
        elif choice == "t":
            console.print("[green]Kalan tüm örnekler kabul edildi.[/green]")
            example["hitl_status"] = "accepted"
            accepted.append(example)
            skip_rest = True
        elif choice == "q":
            console.print("[red]İnceleme sonlandırıldı.[/red]")
            # Kalan örnekleri "unreviewed" olarak işaretle
            for remaining in auto_passed[i + 1:]:
                remaining["hitl_status"] = "unreviewed"
                accepted.append(remaining)  # Varsayılan kabul
            break

    return accepted, rejected


# ─── Veri Seti Bölme ───────────────────────────────────────

def split_dataset(
    examples: list[dict],
    test_ratio: float = 0.10,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, list]:
    """Veri setini train/val/test olarak böl (stratified by category)."""
    import random as rng
    rng.seed(seed)

    # Kategorilere göre grupla
    by_category = {}
    for ex in examples:
        cat = ex.get("category", "other")
        by_category.setdefault(cat, []).append(ex)

    train, val, test = [], [], []

    for cat, cat_examples in by_category.items():
        rng.shuffle(cat_examples)
        n = len(cat_examples)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        test.extend(cat_examples[:n_test])
        val.extend(cat_examples[n_test:n_test + n_val])
        train.extend(cat_examples[n_test + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return {"train": train, "validation": val, "test": test}


# ─── Ana Fonksiyon ─────────────────────────────────────────

def run_filtering(
    input_path: str,
    output_dir: str,
    config_path: str = "configs/config.yaml",
    auto_only: bool = False,
):
    """Filtreleme pipeline'ını çalıştır."""
    config = load_config(config_path)

    # Seed'leri de yükle
    seed_path = "data/seeds/seed_examples.jsonl"
    seeds = []
    if os.path.exists(seed_path):
        with jsonlines.open(seed_path) as reader:
            seeds = list(reader)
            for s in seeds:
                s["source"] = "manual_seed"
                s["hitl_status"] = "accepted"

    # Generated örnekleri yükle
    with jsonlines.open(input_path) as reader:
        generated = list(reader)

    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print(f"[bold]Human-in-the-Loop Filtreleme[/bold]")
    console.print(f"[bold]{'='*60}[/bold]")
    console.print(f"  Seed örnekleri:   {len(seeds)}")
    console.print(f"  Üretilen örnekler: {len(generated)}")

    # Otomatik filtreler
    auto_passed, auto_failed = [], []
    for ex in generated:
        passed, issues = run_auto_filters(ex)
        if passed:
            auto_passed.append(ex)
        else:
            ex["auto_filter_issues"] = issues
            auto_failed.append(ex)

    console.print(f"\n  [green]Otomatik filtre geçen:[/green]  {len(auto_passed)}")
    console.print(f"  [red]Otomatik reddedilen:[/red]  {len(auto_failed)}")

    rejection_rate = len(auto_failed) / len(generated) if generated else 0
    console.print(f"  Otomatik red oranı: {rejection_rate:.1%}")

    if auto_only:
        accepted = auto_passed
        rejected = auto_failed
        console.print("\n[yellow]--auto-only: İnteraktif inceleme atlandı.[/yellow]")
    else:
        accepted, manual_rejected = interactive_review(
            generated, auto_passed, auto_failed
        )
        rejected = auto_failed + manual_rejected

    # Seed'leri ekle
    all_accepted = seeds + accepted

    # Final istatistikler
    total_rejection = len(rejected) / len(generated) if generated else 0
    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print(f"[bold]SONUÇLAR[/bold]")
    console.print(f"  Toplam kabul:     {len(all_accepted)}")
    console.print(f"  Toplam red:       {len(rejected)}")
    console.print(f"  Red oranı:        {total_rejection:.1%}")
    console.print(f"[bold]{'='*60}[/bold]")

    # Veri setini böl
    splits = split_dataset(
        all_accepted,
        test_ratio=config["data"]["test_split"],
        val_ratio=config["data"]["val_split"],
    )

    # Kaydet
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data in splits.items():
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        with jsonlines.open(path, mode="w") as writer:
            writer.write_all(split_data)
        console.print(f"  [green]{split_name}:[/green] {len(split_data)} örnek → {path}")

    # Reddedilenleri de kaydet (analiz için)
    rejected_path = os.path.join(output_dir, "rejected.jsonl")
    with jsonlines.open(rejected_path, mode="w") as writer:
        writer.write_all(rejected)
    console.print(f"  [red]rejected:[/red] {len(rejected)} örnek → {rejected_path}")

    # İstatistik özeti
    stats = {
        "total_generated": len(generated),
        "total_seeds": len(seeds),
        "auto_passed": len(auto_passed),
        "auto_failed": len(auto_failed),
        "final_accepted": len(all_accepted),
        "final_rejected": len(rejected),
        "rejection_rate": round(total_rejection, 4),
        "splits": {k: len(v) for k, v in splits.items()},
    }
    stats_path = os.path.join(output_dir, "filter_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    console.print(f"  [cyan]stats:[/cyan] → {stats_path}")

    return splits


# ─── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Human-in-the-Loop veri seti filtreleme"
    )
    parser.add_argument(
        "--input", type=str, default="data/generated_raw.jsonl",
        help="Self-Instruct çıktı dosyası",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/splits",
        help="Filtrelenmiş veri seti çıktı klasörü",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Config dosyası",
    )
    parser.add_argument(
        "--auto-only", action="store_true",
        help="Sadece otomatik filtreler (interaktif inceleme yok)",
    )
    args = parser.parse_args()

    run_filtering(
        input_path=args.input,
        output_dir=args.output_dir,
        config_path=args.config,
        auto_only=args.auto_only,
    )
