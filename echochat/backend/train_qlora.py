import inspect
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# CHANGE THIS if needed
DEFAULT_MODEL = "mistralai/Mistral-7B-v0.1"
LOW_VRAM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CUDA_AVAILABLE = torch.cuda.is_available()
VRAM_GB = (
    torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if CUDA_AVAILABLE
    else 0.0
)
LOW_VRAM = (not CUDA_AVAILABLE) or (VRAM_GB < 8)
try:
    import bitsandbytes as _bnb  # noqa: F401

    BNB_AVAILABLE = True
    BNB_ERROR = None
except Exception as exc:
    BNB_AVAILABLE = False
    BNB_ERROR = exc
USE_4BIT = CUDA_AVAILABLE and BNB_AVAILABLE


def pick_base_model() -> str:
    override = os.getenv("ECHOCHAT_BASE_MODEL")
    if override:
        return override
    if LOW_VRAM:
        print(
            f"Detected {VRAM_GB:.1f} GB VRAM or no CUDA. "
            f"Using smaller base model: {LOW_VRAM_MODEL}"
        )
        return LOW_VRAM_MODEL
    return DEFAULT_MODEL


BASE_MODEL = pick_base_model()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "training_data.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "echobot-lora"

# 4-bit quantization config (QLoRA)
bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
elif CUDA_AVAILABLE and not BNB_AVAILABLE:
    print(
        "bitsandbytes is unavailable; disabling 4-bit quantization. "
        f"Reason: {BNB_ERROR}"
    )


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token
        else:
            raise ValueError("Tokenizer is missing pad/eos/unk tokens.")
    tok.padding_side = "right"
    return tok


def load_model(model_id: str, use_4bit: bool):
    device_map = "auto" if CUDA_AVAILABLE else {"": "cpu"}
    quant_config = bnb_config if use_4bit else None
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map=device_map,
        torch_dtype=torch.float16 if CUDA_AVAILABLE else torch.float32,
    )


def load_model_and_tokenizer(model_id: str):
    tokenizer = load_tokenizer(model_id)
    if USE_4BIT:
        try:
            model = load_model(model_id, use_4bit=True)
            return model_id, tokenizer, model, True
        except Exception as exc:
            print(
                "4-bit model load failed; retrying without quantization. "
                f"Reason: {exc}"
            )
    try:
        model = load_model(model_id, use_4bit=False)
        return model_id, tokenizer, model, False
    except Exception:
        if model_id != LOW_VRAM_MODEL:
            print(
                f"Falling back to smaller model: {LOW_VRAM_MODEL}"
            )
            return load_model_and_tokenizer(LOW_VRAM_MODEL)
        raise

ACTIVE_MODEL, tokenizer, model, USING_4BIT = load_model_and_tokenizer(BASE_MODEL)
if ACTIVE_MODEL != BASE_MODEL:
    print(f"Using base model: {ACTIVE_MODEL}")

# LoRA configuration (safe for limited VRAM)
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

if USING_4BIT:
    model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
if LOW_VRAM:
    # Reduce memory pressure when VRAM is limited.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

# Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Training data not found: {DATA_PATH}. Expected file under echochat/data/"
    )

dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")
missing_fields = [
    field for field in ("instruction", "input", "output")
    if field not in dataset.column_names
]
if missing_fields:
    print(
        f"Warning: dataset is missing fields {missing_fields}. "
        "Missing fields will be treated as empty strings."
    )


def format_prompt(example):
    return (
        f"### Instruction:\n{example.get('instruction', '')}\n\n"
        f"### Input:\n{example.get('input', '')}\n\n"
        f"### Response:\n{example.get('output', '')}"
    )


# Training arguments (tune only if you know what you are doing)
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=CUDA_AVAILABLE,
    bf16=False,
    gradient_checkpointing=LOW_VRAM,
    logging_steps=20,
    save_steps=500,
    num_train_epochs=3,
    optim="paged_adamw_8bit" if USING_4BIT else "adamw_torch",
    report_to="none"
)

sft_kwargs = dict(
    model=model,
    train_dataset=dataset,
    formatting_func=format_prompt,
    args=training_args,
)
sig = inspect.signature(SFTTrainer.__init__)
if "processing_class" in sig.parameters:
    sft_kwargs["processing_class"] = tokenizer
elif "tokenizer" in sig.parameters:
    sft_kwargs["tokenizer"] = tokenizer
else:
    raise RuntimeError(
        "SFTTrainer signature does not accept 'processing_class' or 'tokenizer'."
    )

trainer = SFTTrainer(**sft_kwargs)

trainer.train()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("Training complete. LoRA adapter saved.")
