import argparse
import inspect
import json
import os
from datetime import datetime
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

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "training_data.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "echobot-lora"

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


def format_prompt(example):
    return (
        f"### Instruction:\n{example.get('instruction', '')}\n\n"
        f"### Input:\n{example.get('input', '')}\n\n"
        f"### Response:\n{example.get('output', '')}"
    )

def pick_base_model(base_model: str | None = None, fast: bool = False) -> str:
    if base_model:
        return base_model
    override = os.getenv("ECHOCHAT_BASE_MODEL")
    if override:
        return override
    if fast or LOW_VRAM:
        print(
            f"Detected {VRAM_GB:.1f} GB VRAM or no CUDA. "
            f"Using smaller base model: {LOW_VRAM_MODEL}"
        )
        return LOW_VRAM_MODEL
    return DEFAULT_MODEL


def train(
    data_path: Path,
    output_dir: Path,
    base_model: str | None = None,
    epochs: float = 3,
    max_steps: int = 0,
    sample_size: int = 0,
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 2e-4,
    fast: bool = False,
    no_amp: bool = False,
) -> None:
    if fast:
        if max_steps <= 0:
            max_steps = 200
        if sample_size <= 0:
            sample_size = 200
        epochs = 1

    base_model = pick_base_model(base_model, fast=fast)
    if "Llama-3.1-8B" in base_model and CUDA_AVAILABLE and VRAM_GB < 8:
        print(
            f"Warning: {base_model} on {VRAM_GB:.1f} GB VRAM may OOM. "
            "If training fails, reduce sample size/steps or use CPU."
        )
    active_model, tokenizer, model, using_4bit = load_model_and_tokenizer(base_model)
    if active_model != base_model:
        print(f"Using base model: {active_model}")

    # LoRA configuration (safe for limited VRAM)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    if using_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    if LOW_VRAM:
        # Reduce memory pressure when VRAM is limited.
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "training_meta.json"
    meta_payload = {
        "base_model": active_model,
        "using_4bit": bool(using_4bit),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    # Load dataset
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}. "
            "Expected file under echochat/data/"
        )

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    missing_fields = [
        field for field in ("instruction", "input", "output")
        if field not in dataset.column_names
    ]
    if missing_fields:
        print(
            f"Warning: dataset is missing fields {missing_fields}. "
            "Missing fields will be treated as empty strings."
        )

    if sample_size > 0 and len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        print(f"Using {len(dataset)} samples for training.")

    max_steps_value = max_steps if max_steps > 0 else -1

    use_fp16 = CUDA_AVAILABLE and not no_amp

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        fp16=use_fp16,
        bf16=False,
        gradient_checkpointing=LOW_VRAM,
        logging_steps=10,
        save_steps=200,
        num_train_epochs=epochs if max_steps_value <= 0 else 1,
        max_steps=max_steps_value,
        optim="paged_adamw_8bit" if using_4bit else "adamw_torch",
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

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Training complete. LoRA adapter saved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EchoChat QLoRA trainer")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        epochs=args.epochs,
        max_steps=args.max_steps,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        fast=args.fast,
        no_amp=args.no_amp,
    )


if __name__ == "__main__":
    main()
