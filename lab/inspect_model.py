"""Đếm tham số từng khối của Qwen3-VL-4B và số module LoRA/tham số huấn luyện của E1–E3.

Dựng model trên thiết bị "meta" (chỉ tải config, không tải trọng số, không cần GPU).
  python -m lab.inspect_model
"""
import re

import torch

from lab.common import MODEL_ID, PROJECTOR_KEYS, lora_pattern


def build():
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    with torch.device("meta"):
        return Qwen3VLForConditionalGeneration(cfg)


def count(model, pred):
    seen, total = set(), 0
    for n, p in model.named_parameters():
        if pred(n) and id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def main():
    from peft import LoraConfig, get_peft_model

    m = build()
    total = count(m, lambda n: True)
    vis = count(m, lambda n: ".visual." in f".{n}")
    merger = count(m, lambda n: "visual.merger" in n)
    deep = count(m, lambda n: "visual.deepstack_merger_list" in n)
    llm = count(m, lambda n: "language_model" in n)
    head = count(m, lambda n: n.startswith("lm_head"))
    print(f"Tổng:                         {total / 1e9:.3f}B")
    print(f"ViT (gồm projector):          {vis / 1e6:.1f}M")
    print(f"  visual.merger:              {merger / 1e6:.1f}M")
    print(f"  visual.deepstack_merger:    {deep / 1e6:.1f}M")
    print(f"  ViT không tính projector:   {(vis - merger - deep) / 1e6:.1f}M")
    print(f"LLM (language_model):         {llm / 1e9:.3f}B")
    print(f"lm_head (riêng, nếu không tied): {head / 1e6:.1f}M  tie_word_embeddings="
          f"{getattr(m.config.text_config, 'tie_word_embeddings', getattr(m.config, 'tie_word_embeddings', None))}")
    print(f"deepstack_visual_indexes: {m.config.vision_config.deepstack_visual_indexes}")

    for name, targets, proj in (("E1", "llm", False), ("E2", "llm", True), ("E3", "llm+vit", True)):
        mm = build()
        pat = lora_pattern(targets)
        matched = [n for n, _ in mm.named_modules() if re.fullmatch(pat, n)]
        with torch.device("meta"):
            pm = get_peft_model(mm, LoraConfig(r=16, lora_alpha=32, target_modules=pat))
        if proj:
            for n, p in pm.named_parameters():
                if any(k in n for k in PROJECTOR_KEYS):
                    p.requires_grad = True
        tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
        print(f"{name}: LoRA modules={len(matched)}  tham số huấn luyện={tr:,}")


if __name__ == "__main__":
    main()
