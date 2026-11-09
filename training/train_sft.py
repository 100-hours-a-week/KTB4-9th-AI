"""Qwen3.5 모델을 문제 생성 데이터로 LoRA 학습한다.

RunPod Pod에서 돌리는 독립 스크립트다. src/를 import하지 않으므로
RunPod 볼륨에 저장한 학습 venv(unsloth, transformers v5)와 build_dataset.py가 만든
train.jsonl, val.jsonl만 있으면 된다.

모델은 HF 캐시(HF_HOME)에 미리 받아둔 스냅샷에서 읽는다.
실행할 때 HF_HUB_OFFLINE=1을 주면 실수로 다시 받는 일이 없다.

실행:
    source /workspace/env.sh
    HF_HUB_OFFLINE=1 python train_sft.py --model Qwen/Qwen3.5-27B \
        --train /workspace/data/train.jsonl --val /workspace/data/val.jsonl \
        --output /workspace/runs/27b-v1
"""

import unsloth  # noqa: F401  transformers보다 먼저 import해야 패치가 적용된다

# isort: split
import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import shutil
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from unsloth import FastLanguageModel

# unsloth가 import될 때 로깅을 먼저 설정해 두므로 force로 덮어써야 INFO가 나온다
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

# Qwen3.5의 Linear 모듈 이름 (2B 스모크에서 확인)
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # full attention (4번째 레이어마다)
    "in_proj_qkv", "in_proj_z", "out_proj",  # 선형 어텐션 (GatedDeltaNet)
    "gate_proj", "up_proj", "down_proj",  # MLP (모든 레이어)
]  # fmt: skip
# in_proj_a, in_proj_b는 출력이 16차원인 게이트라 rank 16 LoRA가 의미 없어 뺐다

IGNORE_INDEX = -100  # loss 계산에서 빠지는 label 값

# 서빙(평가, 배치)과 같은 프롬프트 모양으로 학습한다. 모델마다 템플릿 기본값이
# 달라서(2B는 thinking off, 27B는 on) 명시한다. 서빙에도 같은 값을 준다
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5 LoRA 학습")

    paths = parser.add_argument_group("경로")
    paths.add_argument("--model", required=True, help="HF 모델 ID 또는 로컬 폴더")
    paths.add_argument("--train", type=Path, required=True, help="학습 jsonl")
    paths.add_argument("--val", type=Path, required=True, help="검증 jsonl")
    paths.add_argument("--output", type=Path, required=True, help="결과 폴더")

    lora = parser.add_argument_group("LoRA")
    lora.add_argument("--rank", type=int, default=16)
    lora.add_argument("--alpha", type=int, default=16)

    train = parser.add_argument_group("학습")
    train.add_argument("--epochs", type=float, default=2)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--batch-size", type=int, default=1, help="GPU당 배치")
    train.add_argument("--grad-accum", type=int, default=8, help="그래디언트 누적")
    train.add_argument(
        "--warmup", type=float, default=0.05, help="1 미만이면 비율, 이상이면 스텝"
    )
    train.add_argument("--max-seq-length", type=int, default=4096)
    train.add_argument(
        "--max-steps", type=int, default=-1, help="스모크용. 양수면 에폭 대신 쓴다"
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--resume", action="store_true", help="output의 마지막 체크포인트부터 이어서"
    )
    train.add_argument(
        "--report-to",
        choices=["none", "wandb"],
        default="none",
        help="학습 기록을 보낼 곳. wandb는 WANDB_API_KEY와 wandb 패키지가 필요하다",
    )

    merge = parser.add_argument_group("병합")
    merge.add_argument("--merge", action="store_true", help="학습 후 bf16 병합 저장")
    merge.add_argument(
        "--merge-dir", type=Path, help="병합 저장 폴더. 없으면 output/merged"
    )

    return parser.parse_args()


def resolve_snapshot(model: str) -> Path:
    """
    모델 ID를 HF 캐시의 스냅샷 폴더로 바꾼다. 로컬 폴더면 그대로 쓴다.

    폴더 경로로 로드하면 병합할 때 unsloth가 Hub에 묻지 않고
    이 폴더의 원본 가중치를 읽는다.

    Raises:
        FileNotFoundError: 캐시에 없거나 가중치 파일이 빠진 경우
    """
    path = Path(model)
    if not path.is_dir():
        # 네트워크 없이 캐시에서만 찾는다. 없으면 받지 않고 예외를 낸다
        path = Path(snapshot_download(model, local_files_only=True))

    index = path / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        missing = sorted(name for name in shards if not (path / name).exists())
        if missing:
            raise FileNotFoundError(f"{path}에 가중치 파일이 없습니다: {missing}")
    elif not list(path.glob("*.safetensors")):
        raise FileNotFoundError(f"{path}에 safetensors 파일이 없습니다.")

    return path


def load_model(snapshot: Path, max_seq_length: int):
    """
    bf16으로 로드한다. Qwen3.5는 멀티모달이라 토크나이저 대신 processor가 온다.

    Returns:
        tuple: (model, processor, tokenizer). tokenizer는 processor 안의
            텍스트 토크나이저로, chat template과 토큰화에 쓴다.
            processor는 어댑터와 같이 저장해 서빙에서 쓴다.
    """
    model, processor = FastLanguageModel.from_pretrained(
        model_name=str(snapshot),
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        raise ValueError("pad 토큰이 없어 배치 패딩을 할 수 없습니다.")

    logger.info(
        "로드: %s | %s | %s | %s",
        type(model).__name__,
        type(processor).__name__,
        type(tokenizer).__name__,
        model.dtype,
    )
    logger.info(
        "eos %r | pad %r | VRAM %.1fGB",
        tokenizer.eos_token,
        tokenizer.pad_token,
        torch.cuda.memory_allocated() / 1024**3,
    )
    return model, processor, tokenizer


def add_lora(model, rank: int, alpha: int, max_seq_length: int, seed: int):
    """
    LoRA를 붙이고 붙은 위치를 확인한다.

    PEFT는 모듈 이름의 끝부분으로 대상을 찾으므로, 비전 인코더에 같은 이름이
    있으면 거기에도 붙을 수 있다. 목록 중 한 곳도 안 붙은 모듈은 이름이 다른
    모델이라는 뜻이다. 둘 다 학습을 시작하기 전에 멈춘다.

    Raises:
        RuntimeError: 비전 인코더에 붙었거나 붙지 않은 대상 모듈이 있는 경우
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0,
        bias="none",
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        max_seq_length=max_seq_length,
    )

    counts: Counter[tuple[str, str]] = Counter()
    for name, module in model.named_modules():
        if getattr(module, "lora_A", None):
            part = "vision" if "visual" in name else "language"
            counts[(part, name.rsplit(".", 1)[-1])] += 1
    for (part, kind), count in sorted(counts.items()):
        logger.info("LoRA %-8s %-12s %d", part, kind, count)

    if any(part == "vision" for part, _ in counts):
        raise RuntimeError("비전 인코더에 LoRA가 붙었습니다. 대상 모듈을 좁히세요.")
    missing = set(TARGET_MODULES) - {kind for _, kind in counts}
    if missing:
        raise RuntimeError(f"LoRA가 붙지 않은 대상 모듈: {sorted(missing)}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "학습 파라미터 %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        trainable / total * 100,
    )
    return model


def to_features(messages: list[dict], tokenizer) -> dict:
    """
    대화 하나를 학습 샘플로 만든다. 정답(assistant) 토큰만 loss에 들어간다.

    프롬프트는 서빙과 같은 모양(add_generation_prompt=True)으로 만들고
    그 토큰 수만큼 labels를 IGNORE_INDEX로 가린다. 이 방식은 프롬프트 토큰이
    전체 토큰의 앞부분과 정확히 같아야 맞으므로 확인한다.

    Raises:
        ValueError: [user, assistant] 대화가 아니거나 토큰 경계가 어긋난 경우
    """
    roles = [message["role"] for message in messages]
    if roles != ["user", "assistant"]:
        raise ValueError(f"[user, assistant] 대화가 아닙니다: {roles}")

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, **CHAT_TEMPLATE_KWARGS
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "프롬프트 토큰이 전체 토큰의 앞부분과 다릅니다. "
            f"프롬프트 끝 {prompt_text[-60:]!r}"
        )

    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def percentile(values: list[int], q: float) -> int:
    """정렬했을 때 q 위치의 값. q=0.95면 p95."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)]


def build_dataset(path: Path, tokenizer, max_seq_length: int):
    """
    jsonl을 학습 샘플 Dataset으로 만들고, 실행 기록에 남길 정보를 같이 돌려준다.

    긴 샘플을 조용히 자르면 정답 끝(<|im_end|>)이 빠져 멈추는 법을 못 배우므로
    max_seq_length를 넘는 샘플이 있으면 멈춘다.

    Raises:
        ValueError: max_seq_length보다 긴 샘플이 있는 경우
    """
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    features = [to_features(row["messages"], tokenizer) for row in rows]

    total = [len(feature["input_ids"]) for feature in features]
    answer = [
        sum(label != IGNORE_INDEX for label in feature["labels"])
        for feature in features
    ]
    info = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "count": len(rows),
        "total_tokens": {
            "p50": statistics.median(total),
            "p95": percentile(total, 0.95),
            "max": max(total),
        },
        "answer_tokens": {
            "p50": statistics.median(answer),
            "p95": percentile(answer, 0.95),
            "max": max(answer),
        },
    }
    logger.info(
        "데이터 %s: %d건 | 전체 토큰 %s | 정답 토큰 %s",
        path.name,
        info["count"],
        info["total_tokens"],
        info["answer_tokens"],
    )
    if info["total_tokens"]["max"] > max_seq_length:
        raise ValueError(
            f"{path.name}의 가장 긴 샘플({info['total_tokens']['max']}토큰)이 "
            f"max_seq_length({max_seq_length})를 넘습니다."
        )
    return Dataset.from_list(features), info


def log_boundary(dataset: Dataset, tokenizer) -> None:
    """첫 샘플에서 가려지는 끝과 학습이 시작되는 곳을 보여준다."""
    sample = dataset[0]
    start = next(i for i, label in enumerate(sample["labels"]) if label != IGNORE_INDEX)
    logger.info(
        "가려지는 끝 %r | 학습 시작 %r | 학습 끝 %r",
        tokenizer.decode(sample["input_ids"][start - 8 : start]),
        tokenizer.decode(sample["labels"][start : start + 8]),
        tokenizer.decode(sample["labels"][-3:]),
    )


def train(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    args: argparse.Namespace,
) -> tuple[Trainer, dict]:
    """
    학습하고 검증 loss를 잰다. 에폭마다 검증하고 체크포인트를 남긴다.

    Returns:
        tuple: (trainer, 결과). 결과는 시간, 스텝, loss, 최대 VRAM으로
            실행 기록에 남긴다.
    """
    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,  # 양수면 에폭 대신 이 스텝만 돈다
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        # v5에는 warmup_ratio가 없다. 1 미만 실수를 주면 비율로 쓴다
        warmup_steps=args.warmup,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        bf16=True,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=args.report_to,
        run_name=args.output.name,  # wandb에서 실행을 구분하는 이름
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # 배치 안에서 길이를 맞출 때 labels는 IGNORE_INDEX로 채운다
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX
        ),
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    # --resume이면 output/checkpoints의 마지막 체크포인트부터 이어서 학습한다
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    elapsed = time.time() - started

    # 스모크(max_steps)는 에폭이 끝나지 않아 자동 검증이 돌지 않으므로 끝에서 잰다
    eval_loss = trainer.evaluate()["eval_loss"]
    train_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    result = {
        "seconds": round(elapsed),
        "steps": trainer.state.global_step,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "eval_loss": eval_loss,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 1),
    }
    logger.info("학습 끝: %s", result)
    return trainer, result


def save_adapter(model, processor, output: Path) -> Path:
    """
    LoRA 어댑터와 processor를 저장한다. vLLM --lora-modules에 이 폴더를 넘긴다.

    Returns:
        Path: 어댑터 폴더
    """
    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    size = sum(f.stat().st_size for f in adapter_dir.iterdir() if f.is_file())
    logger.info("어댑터 저장: %s (%.2fGB)", adapter_dir, size / 1024**3)
    return adapter_dir


def package_versions() -> dict[str, str]:
    """결과를 재현할 때 필요한 주요 패키지 버전."""
    names = ["torch", "transformers", "trl", "peft", "unsloth", "unsloth_zoo"]
    return {name: version(name) for name in names}


def write_run_config(
    path: Path,
    args: argparse.Namespace,
    snapshot: Path,
    data: dict,
    result: dict,
    log_history: list[dict],
) -> None:
    """
    이 실행을 나중에 비교·재현할 수 있도록 설정과 결과를 한 파일에 남긴다.

    어댑터를 저장한 뒤에 부른다. 저장에 실패한 실행이 성공처럼 기록되지 않게 한다.
    """
    config = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model": {"id": args.model, "snapshot": str(snapshot)},
        "target_modules": TARGET_MODULES,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {**package_versions(), "cuda": torch.version.cuda},
        "data": data,
        "result": result,
        "log_history": log_history,  # 스텝별 loss, 학습률, 에폭별 검증 loss
    }
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("실행 기록: %s", path)


def weight_keys(directory: Path) -> set[str]:
    """폴더의 safetensors 가중치 이름. 헤더만 읽으므로 큰 모델도 빠르다."""
    keys: set[str] = set()
    for path in directory.glob("*.safetensors"):
        with safe_open(path, "pt") as f:
            keys.update(f.keys())
    return keys


def merge(model, processor, snapshot: Path, merge_dir: Path) -> dict:
    """
    LoRA를 베이스에 더한 bf16 모델을 저장한다.

    unsloth는 베이스 가중치를 merge_dir로 통째로 복사한 뒤 LoRA를 더한다.
    그래서 베이스 크기만큼 빈 공간이 있어야 한다. bf16 반올림으로 작은 LoRA
    변화량 일부가 사라지므로 평가에는 어댑터 서빙이 더 정확하다.

    Returns:
        dict: 실행 기록에 남길 병합 정보

    Raises:
        RuntimeError: 공간이 모자라거나 병합 결과가 베이스와 모양이 다른 경우
    """
    # 스냅샷 파일은 blob을 가리키는 링크라 stat이 실제 크기를 준다
    needed = sum(path.stat().st_size for path in snapshot.glob("*.safetensors"))
    merge_dir.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(merge_dir.parent).free
    logger.info(
        "병합 → %s | 필요 %.0fGB | 남은 공간 %.0fGB",
        merge_dir,
        needed / 1024**3,
        free / 1024**3,
    )
    if free < needed:
        raise RuntimeError("병합할 공간이 모자랍니다. --merge-dir를 바꾸세요.")

    started = time.time()
    model.save_pretrained_merged(str(merge_dir), processor, save_method="merged_16bit")

    # 베이스의 샘플링 기본값과 eos 목록을 유지한다. 병합은 새로 만들어 쓰기 때문이다
    base_generation = snapshot / "generation_config.json"
    if base_generation.exists():
        shutil.copy2(base_generation, merge_dir / "generation_config.json")

    base_keys, merged_keys = weight_keys(snapshot), weight_keys(merge_dir)
    if merged_keys != base_keys:
        raise RuntimeError(
            "병합 가중치가 베이스와 다릅니다. "
            f"베이스에만: {sorted(base_keys - merged_keys)[:5]} "
            f"병합에만: {sorted(merged_keys - base_keys)[:5]}"
        )

    info = {
        "dir": str(merge_dir),
        "seconds": round(time.time() - started),
        "weights": len(merged_keys),
        "generation_config_from_base": base_generation.exists(),
    }
    logger.info("병합 끝: %s", info)
    return info


def add_to_run_config(path: Path, key: str, value: dict) -> None:
    """이미 쓴 실행 기록에 항목 하나를 더한다."""
    config = json.loads(path.read_text(encoding="utf-8"))
    config[key] = value
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def check_report_to(report_to: str) -> None:
    """
    wandb에 보낼 준비가 됐는지 모델을 올리기 전에 확인한다.

    Raises:
        RuntimeError: wandb 패키지나 WANDB_API_KEY가 없는 경우
    """
    if report_to != "wandb":
        return
    if importlib.util.find_spec("wandb") is None:
        raise RuntimeError("wandb 패키지가 없습니다. uv pip install wandb")
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY 환경변수가 없습니다.")


def main() -> None:
    args = parse_args()
    check_report_to(args.report_to)
    snapshot = resolve_snapshot(args.model)
    logger.info("모델 %s → %s", args.model, snapshot)

    model, processor, tokenizer = load_model(snapshot, args.max_seq_length)
    model = add_lora(model, args.rank, args.alpha, args.max_seq_length, args.seed)

    train_dataset, train_info = build_dataset(
        args.train, tokenizer, args.max_seq_length
    )
    val_dataset, val_info = build_dataset(args.val, tokenizer, args.max_seq_length)
    log_boundary(train_dataset, tokenizer)

    trainer, result = train(model, tokenizer, train_dataset, val_dataset, args)

    save_adapter(model, processor, args.output)
    run_config = args.output / "run_config.json"
    write_run_config(
        run_config,
        args,
        snapshot,
        {"train": train_info, "val": val_info},
        result,
        trainer.state.log_history,
    )

    # 병합이 실패해도 학습 결과(어댑터와 기록)는 이미 남아 있다
    if args.merge:
        merge_dir = args.merge_dir or args.output / "merged"
        add_to_run_config(
            run_config, "merge", merge(model, processor, snapshot, merge_dir)
        )


if __name__ == "__main__":
    main()
