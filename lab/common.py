"""Hàm dùng chung: nhãn JSON, prompt, nạp model 4-bit, mask loss, setup LoRA, suy luận."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.labcommon import CLASSES, read_jsonl  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
IM_END = "<|im_end|>"

# --- Bước 2: nhãn YOLO -> JSON -------------------------------------------------

def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    """YOLO (xc, yc, w, h chuẩn hoá [0,1]) -> {"label", "bbox_2d": [x1,y1,x2,y2]} số nguyên."""
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    box = [(xc - w / 2) * sx, (yc - h / 2) * sy, (xc + w / 2) * sx, (yc + h / 2) * sy]
    lim = [sx, sy, sx, sy]
    box = [int(min(max(round(v), 0), m)) for v, m in zip(box, lim)]
    return {"label": label, "bbox_2d": box}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))  # trên->dưới, trái->phải
    return json.dumps(boxes, separators=(",", ":"))


def make_prompt(coord="norm1000", width=640, height=360):
    if coord == "norm1000":
        scale = ("Coordinates are integers on a 0-1000 scale relative to the image "
                 "(0 = left/top edge, 1000 = right/bottom edge).")
    else:
        scale = (f"Coordinates are integer pixel positions in the {width}x{height} image "
                 f"(x from 0 to {width}, y from 0 to {height}).")
    return ("Detect all road damage in this dashcam image. "
            "Classes: pothole, crack, manhole. "
            'Output ONLY a JSON list, no other text: [{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}]. '
            f"{scale} (x1, y1) is the top-left corner and (x2, y2) the bottom-right corner. "
            "If there is no damage, output [].")


def load_split(split, data_dir="data", limit=None):
    rows = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    return rows[:limit] if limit else rows


def open_image(entry, data_dir="data"):
    from PIL import Image
    return Image.open(Path(data_dir) / entry["image_path"]).convert("RGB")


# --- Bước 3: nạp model 4-bit ---------------------------------------------------

def load_model(model_id=MODEL_ID):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    # ViT + projector phải ở fp16 (không lượng tử) thì mới huấn luyện đầy đủ projector được
    w = model.model.visual.merger.linear_fc1.weight
    assert w.dtype == torch.float16, f"visual.merger bị lượng tử hoá ({w.dtype})"
    return model, processor


def chat_prompt(processor, prompt):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# --- Bước 4: ví dụ huấn luyện + mask loss --------------------------------------

def encode_example(processor, image, prompt, answer):
    """Trả dict tensor [1, L] + labels; chỉ phần trả lời (kể cả <|im_end|>) được tính loss."""
    ptext = chat_prompt(processor, prompt)
    full = ptext + answer + IM_END + "\n"
    p = processor(text=[ptext], images=[image], return_tensors="pt")
    f = processor(text=[full], images=[image], return_tensors="pt")
    n = p["input_ids"].shape[1]
    if not (f["input_ids"][0, :n] == p["input_ids"][0]).all():
        raise ValueError("prompt không phải tiền tố của chuỗi đầy đủ")
    out = dict(f)
    labels = f["input_ids"].clone()
    labels[:, :n] = -100
    out["labels"] = labels
    return out


def check_template(processor, prompt, answer):
    """So chuỗi tự ghép với chat template chính thức (in cảnh báo nếu khác)."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    official = processor.apply_chat_template(msgs, tokenize=False)
    mine = chat_prompt(processor, prompt) + answer + IM_END + "\n"
    if official.strip() != mine.strip():
        print("CẢNH BÁO: chuỗi tự ghép khác template chính thức\n--- official ---\n"
              f"{official}\n--- mine ---\n{mine}")
        return False
    return True


def collate(features, pad_id):
    import torch
    out = {}
    L = max(f["input_ids"].shape[1] for f in features)
    for k in features[0]:
        if k in ("pixel_values", "image_grid_thw"):
            out[k] = torch.cat([f[k] for f in features])
            continue
        val = -100 if k == "labels" else (pad_id if k == "input_ids" else 0)
        rows = []
        for f in features:
            t = f[k]
            pad = torch.full((1, L - t.shape[1]), val, dtype=t.dtype)
            rows.append(torch.cat([t, pad], dim=1))  # đệm phải khi huấn luyện
        out[k] = torch.cat(rows)
    return out


# --- Bước 5: setup LoRA / đóng băng --------------------------------------------

LLM_RE = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_RE = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc(1|2))"
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")


def lora_pattern(targets):
    return LLM_RE if targets == "llm" else f"{LLM_RE}|{VIT_RE}"


def apply_setup(model, targets="llm", train_projector=False, r=16, alpha=32, dropout=0.05,
                gradient_checkpointing=True, verbose=True):
    import torch
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    pattern = lora_pattern(targets)
    matched = [n for n, _ in model.named_modules() if re.fullmatch(pattern, n)]
    peft_model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=pattern, bias="none"))
    n_proj = 0
    if train_projector:
        for n, p in peft_model.named_parameters():
            if any(k in n for k in PROJECTOR_KEYS):
                p.data = p.data.float()  # fp32 để AdamW/GradScaler cập nhật ổn định
                p.requires_grad = True
                n_proj += p.numel()
    for p in peft_model.parameters():  # GradScaler không unscale được gradient fp16
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if verbose:
        print(f"LoRA modules khớp: {len(matched)} | projector mở: {n_proj:,} | "
              f"tham số huấn luyện: {trainable:,}")
    return peft_model, {"matched_modules": len(matched), "projector_params": n_proj,
                        "trainable_params": trainable}


def trainable_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, path):
    import torch
    sd = torch.load(path, map_location="cpu")
    res = model.load_state_dict(sd, strict=False)
    if res.unexpected_keys:
        raise RuntimeError(f"khoá không khớp model (setup khác lúc train?): {res.unexpected_keys[:5]}")
    missing = set(sd) & set(res.missing_keys)
    if missing:
        raise RuntimeError(f"không nạp được: {list(missing)[:5]}")
    print(f"đã nạp {len(sd)} tensor huấn luyện từ {path}")


# --- Bước 3/7: suy luận ----------------------------------------------------------

def predict(model, processor, entries, prompt, out_path, data_dir="data",
            batch_size=4, max_new_tokens=384):
    """Sinh output thô cho từng ảnh, ghi nối tiếp vào out_path (chạy lại thì làm tiếp phần còn thiếu)."""
    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["image_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [e for e in entries if e["image_id"] not in done]
    print(f"suy luận {len(todo)} ảnh (đã có {len(done)}) -> {out_path}")
    processor.tokenizer.padding_side = "left"
    ptext = chat_prompt(processor, prompt)
    model.eval()
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            imgs = [open_image(e, data_dir) for e in batch]
            inputs = processor(text=[ptext] * len(batch), images=imgs, padding=True,
                               return_tensors="pt").to(model.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     use_cache=True)
            raws = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            for e, raw in zip(batch, raws):
                f.write(json.dumps({"image_id": e["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            if (i // batch_size) % 10 == 0:
                print(f"  {i + len(batch)}/{len(todo)}  ví dụ: {raws[0][:120]!r}", flush=True)
    return out_path


__all__ = ["CLASSES", "MODEL_ID", "yolo_to_box", "objects_to_answer", "make_prompt", "load_split",
           "open_image", "load_model", "chat_prompt", "encode_example", "check_template", "collate",
           "apply_setup", "trainable_state", "load_trainable", "predict"]
