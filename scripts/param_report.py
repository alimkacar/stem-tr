"""
param_report.py — QLoRA eğitilebilir parametre azaltma raporu
=============================================================
Turkish-Llama-8B-Instruct + LoRA (r, hedef modüller config'ten) için
eğitilebilir / toplam parametre oranını ANALİTİK olarak hesaplar.
GPU veya model indirmeye gerek yoktur; sonuç config ile doğrudan üretilir.

İsteğe bağlı doğrulama: transformers + peft + model erişimi varsa
`--verify` ile gerçek `print_trainable_parameters()` çıktısı da alınır.

Kullanım:
    python scripts/param_report.py --config configs/config.yaml
    python scripts/param_report.py --config configs/config.yaml --verify   # gerçek modelle
"""
import argparse
import json
import os

import yaml

# Llama-3 8B mimarisi (ytu-ce-cosmos/Turkish-Llama-8B-Instruct temeli)
LLAMA3_8B = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,     # GQA
    "head_dim": 128,
    "vocab_size": 128256,
    "total_params": 8_030_261_248,  # ~8.03B (resmi Llama-3-8B parametre sayısı)
}


def module_in_out(name: str, arch: dict) -> tuple[int, int]:
    """Bir lineer modülün (giriş, çıkış) boyutlarını döndür."""
    h = arch["hidden_size"]
    inter = arch["intermediate_size"]
    q_out = arch["num_attention_heads"] * arch["head_dim"]      # 4096
    kv_out = arch["num_key_value_heads"] * arch["head_dim"]     # 1024
    table = {
        "q_proj": (h, q_out),
        "k_proj": (h, kv_out),
        "v_proj": (h, kv_out),
        "o_proj": (q_out, h),
        "gate_proj": (h, inter),
        "up_proj": (h, inter),
        "down_proj": (inter, h),
    }
    if name not in table:
        raise ValueError(f"Bilinmeyen hedef modül: {name}")
    return table[name]


def compute(config: dict, arch: dict = LLAMA3_8B) -> dict:
    r = config["training"]["lora"]["r"]
    targets = config["training"]["lora"]["target_modules"]
    layers = arch["num_hidden_layers"]

    per_module = {}
    trainable = 0
    for m in targets:
        i, o = module_in_out(m, arch)
        # LoRA: A (r×in) + B (out×r) => r*(in+out) parametre
        p = r * (i + o)
        per_module[m] = p
        trainable += p * layers

    total = arch["total_params"]
    pct = 100.0 * trainable / total
    return {
        "base_model": config["training"]["base_model"],
        "lora_rank": r,
        "target_modules": targets,
        "num_layers": layers,
        "lora_params_per_module_per_layer": per_module,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(pct, 4),
        "reduction_pct": round(100.0 - pct, 4),
    }


def verify_with_peft(config: dict) -> dict | None:
    """Gerçek modeli yükleyip PEFT ile doğrula (GPU/model gerektirir)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("[WARN] transformers/peft yok; doğrulama atlandı.")
        return None
    q = config["training"]["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["training"]["base_model"], quantization_config=bnb, device_map="auto"
    )
    lc = config["training"]["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lc["r"], lora_alpha=lc["lora_alpha"], lora_dropout=lc["lora_dropout"],
        target_modules=lc["target_modules"], bias=lc["bias"], task_type=TaskType.CAUSAL_LM,
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": trainable, "total_params_counted": total,
            "trainable_pct": round(100.0 * trainable / total, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--verify", action="store_true", help="Gerçek modelle doğrula (GPU gerekir)")
    ap.add_argument("--out", default="outputs/param_report.json")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rep = compute(config)

    print("=" * 62)
    print("QLoRA Eğitilebilir Parametre Raporu (analitik)")
    print("=" * 62)
    print(f"  Temel model      : {rep['base_model']}")
    print(f"  LoRA rank (r)    : {rep['lora_rank']}")
    print(f"  Hedef modüller   : {', '.join(rep['target_modules'])}")
    print(f"  Katman sayısı    : {rep['num_layers']}")
    print(f"  Eğitilebilir     : {rep['trainable_params']:,}")
    print(f"  Toplam           : {rep['total_params']:,}")
    print(f"  Eğitilebilir %   : {rep['trainable_pct']}%")
    print(f"  AZALTMA          : %{rep['reduction_pct']}  (eğitilebilir parametre)")
    print("=" * 62)

    if args.verify:
        v = verify_with_peft(config)
        if v:
            rep["peft_verification"] = v
            print(f"[PEFT] Gerçek: eğitilebilir {v['trainable_params']:,} / "
                  f"{v['total_params_counted']:,} = {v['trainable_pct']}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"[OK] Rapor kaydedildi: {args.out}")


if __name__ == "__main__":
    main()
