#!/usr/bin/env python3
"""
Build GSM8K misleading-hint datasets (train/eval) and write a preview artifact.

This is intentionally separate from training so you can:
- Build once (expensive: OpenAI calls)
- Inspect `dataset_preview.md` immediately
- Reuse `train_dataset.jsonl` / `eval_dataset.jsonl` across training reruns
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow `python GSM/build_gsm_hint_dataset.py` from repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gsm_hint_dataset import build_tasks_with_hints, write_dataset_preview, write_tasks_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Directory to write dataset files.")
    ap.add_argument("--openai_hint_model", default="gpt-5-mini")
    ap.add_argument("--hint_cache_path", default="", help="JSONL cache for hint generations (recommended).")
    ap.add_argument("--timeout_s", type=float, default=60.0)

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_train_tasks", type=int, default=256)
    ap.add_argument("--n_val_tasks", type=int, default=0, help="Optional validation set size (sampled from train split).")
    ap.add_argument("--n_eval_tasks", type=int, default=128)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="train")
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--user_answer_correct_frac", type=float, default=0.5, help="Fraction of examples where user_answer is the true answer.")
    ap.add_argument("--preview_n", type=int, default=5)
    ap.add_argument("--num_shards", type=int, default=1, help="If >1, build only a shard of the dataset.")
    ap.add_argument("--shard_idx", type=int, default=0, help="Which shard index to build (0-based).")

    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    hint_cache = args.hint_cache_path.strip()
    if not hint_cache:
        hint_cache = os.path.join(out_dir, "hint_cache.jsonl")
    hint_cache = os.path.abspath(hint_cache)

    train_tasks = build_tasks_with_hints(
        n=int(args.n_train_tasks),
        seed=int(args.seed),
        split=str(args.train_split),
        openai_model=str(args.openai_hint_model),
        hint_cache_path=hint_cache,
        timeout_s=float(args.timeout_s),
        user_answer_correct_frac=float(args.user_answer_correct_frac),
        shard_idx=int(args.shard_idx),
        num_shards=int(args.num_shards),
    )
    val_tasks = []
    if int(args.n_val_tasks) > 0:
        # GSM8K has only train/test. We create a "val" by sampling from train with a different seed.
        val_tasks = build_tasks_with_hints(
            n=int(args.n_val_tasks),
            seed=int(args.seed) + 2025,
            split=str(args.val_split),
            openai_model=str(args.openai_hint_model),
            hint_cache_path=hint_cache,
            timeout_s=float(args.timeout_s),
            user_answer_correct_frac=float(args.user_answer_correct_frac),
            shard_idx=int(args.shard_idx),
            num_shards=int(args.num_shards),
        )
    eval_tasks = build_tasks_with_hints(
        n=int(args.n_eval_tasks),
        seed=int(args.seed) + 1337,
        split=str(args.eval_split),
        openai_model=str(args.openai_hint_model),
        hint_cache_path=hint_cache,
        timeout_s=float(args.timeout_s),
        user_answer_correct_frac=float(args.user_answer_correct_frac),
        shard_idx=int(args.shard_idx),
        num_shards=int(args.num_shards),
    )

    suffix = ""
    if int(args.num_shards) > 1:
        suffix = f"_shard_{int(args.shard_idx)}_of_{int(args.num_shards)}"
    train_path = os.path.join(out_dir, f"train_dataset{suffix}.jsonl")
    val_path = os.path.join(out_dir, f"val_dataset{suffix}.jsonl")
    eval_path = os.path.join(out_dir, f"eval_dataset{suffix}.jsonl")
    write_tasks_jsonl(train_path, train_tasks)
    if val_tasks:
        write_tasks_jsonl(val_path, val_tasks)
    write_tasks_jsonl(eval_path, eval_tasks)

    previews = write_dataset_preview(
        out_dir=out_dir,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        val_tasks=val_tasks if val_tasks else None,
        n=int(args.preview_n),
    )

    print("[dataset] wrote", train_path, flush=True)
    if val_tasks:
        print("[dataset] wrote", val_path, flush=True)
    print("[dataset] wrote", eval_path, flush=True)
    print("[dataset] wrote", previews["preview_md"], flush=True)
    print("[dataset] wrote", previews["preview_json"], flush=True)
    print("[dataset] hint cache", hint_cache, flush=True)


if __name__ == "__main__":
    main()


