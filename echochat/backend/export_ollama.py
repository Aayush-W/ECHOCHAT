import argparse
import subprocess
from pathlib import Path


def validate_adapter_dir(adapter_dir: Path) -> None:
    if (adapter_dir / "adapter_config.json").exists():
        return
    if list(adapter_dir.glob("*.safetensors")):
        return
    if list(adapter_dir.glob("*.bin")):
        return
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def build_modelfile(base_model: str, adapter_dir: Path) -> str:
    adapter_dir = adapter_dir.resolve()
    return "\n".join(
        [
            f"FROM {base_model}",
            f"ADAPTER {adapter_dir}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an Ollama model from a LoRA adapter.")
    parser.add_argument("--adapter-dir", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--base-model", required=True, help="Ollama base model name (e.g., mistral)")
    parser.add_argument("--model-name", required=True, help="New Ollama model name to create")
    parser.add_argument("--modelfile-dir", default=None, help="Directory to write Modelfile")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    validate_adapter_dir(adapter_dir)

    modelfile_dir = Path(args.modelfile_dir) if args.modelfile_dir else adapter_dir
    modelfile_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = modelfile_dir / f"Modelfile.{args.model_name}"

    modelfile_path.write_text(
        build_modelfile(args.base_model, adapter_dir),
        encoding="utf-8",
    )

    cmd = ["ollama", "create", args.model_name, "-f", str(modelfile_path)]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"ollama create failed with code {result.returncode}")

    print(f"Created Ollama model: {args.model_name}")


if __name__ == "__main__":
    main()
