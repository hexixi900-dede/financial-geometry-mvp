"""Raw Qwen2.5-VL baseline for the Phase 5 natural-chart pilot.

The runner reads image/question inputs only. It writes one JSONL record after
each answer so an interrupted GPU run can be resumed safely.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gpu_is_idle(max_used_memory_mib: int = 1024) -> bool:
    process_check = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    memory_check = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if process_check.returncode or memory_check.returncode:
        return False
    processes = [line for line in process_check.stdout.splitlines() if line.strip()]
    used = [int(line.strip()) for line in memory_check.stdout.splitlines() if line.strip()]
    return not processes and bool(used) and all(value <= max_used_memory_mib for value in used)


def parse_numeric(text: str) -> float | None:
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        value = float(matches[-1].replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def conversation(row: dict[str, str], min_pixels: int, max_pixels: int) -> list[dict[str, Any]]:
    question = row["question"].strip()
    prompt = (
        "Answer the financial chart question from the image. Return only one "
        "numeric final answer in the units requested, with no explanation.\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": "Read the chart carefully and return only the final numeric answer."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Path(row["image_path"]).resolve().as_uri(),
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed[row["sample_id"]] = row
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=451584)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if not gpu_is_idle():
        raise RuntimeError("GPU is not idle; refusing to load the VLM")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite completed output: {args.output}")

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    rows = read_csv(args.inputs)
    progress_path = args.output.with_suffix(".progress.jsonl")
    completed = load_completed(progress_path)
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    ).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as progress:
        for index, row in enumerate(rows, 1):
            if row["sample_id"] in completed:
                continue
            messages = conversation(row, args.min_pixels, args.max_pixels)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            kwargs: dict[str, Any] = {
                "text": [text],
                "images": image_inputs,
                "padding": True,
                "return_tensors": "pt",
            }
            if video_inputs:
                kwargs["videos"] = video_inputs
            inputs = processor(**kwargs).to(args.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            generated = output_ids[:, inputs["input_ids"].shape[1] :]
            answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            numeric = parse_numeric(answer)
            record = {
                "sample_id": row["sample_id"],
                "image_id": row["image_id"],
                "question": row["question"],
                "operation": row["operation"],
                "status": "success" if numeric is not None else "numeric_parse_failed",
                "raw_vlm_text": answer,
                "prediction": "" if numeric is None else numeric,
            }
            progress.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.flush()
            completed[row["sample_id"]] = record
            print(json.dumps({"processed": index, "total": len(rows), "status": record["status"]}), flush=True)

    ordered = [completed[row["sample_id"]] for row in rows if row["sample_id"] in completed]
    write_csv(args.output, ordered)
    print(json.dumps({"rows": len(ordered), "output": str(args.output)}))


if __name__ == "__main__":
    main()
