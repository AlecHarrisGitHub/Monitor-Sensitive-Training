#!/bin/bash
#SBATCH -J mst_gsm_poc
# NOTE: Cluster-specific settings below. Adjust for your environment.
# (On PACE you likely want to set an allocation and a GPU partition.)
# #SBATCH -A <your_allocation>
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH -t 08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# ---- Secrets / tokens ----
# IMPORTANT: Do not hardcode secrets in git.
# Set these in your environment before submitting (or use your cluster's secrets tooling).
#
# If your HuggingFace model is gated, set one of:
#   export HUGGING_FACE_HUB_TOKEN=...
#   export HF_TOKEN=...

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment before submit.}"

# ---- Caches (recommended on PACE) ----
export HF_HOME="${HOME}/scratch/hf_cache"
export HF_DATASETS_CACHE="${HOME}/scratch/hf_datasets_cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

# ---- Conda env bootstrap (mirrors CTG style) ----
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

ENVNAME=mst-gsm-poc-ppo0101
SCRATCH_BASE="${SCRATCH_BASE:-${HOME}/scratch}"
mkdir -p "${SCRATCH_BASE}"

export CONDA_PKGS_DIRS="${SCRATCH_BASE}/conda_pkgs"
mkdir -p "${CONDA_PKGS_DIRS}"

ENV_PREFIX="${SCRATCH_BASE}/conda_envs/${ENVNAME}"
mkdir -p "$(dirname "${ENV_PREFIX}")"
if [[ ! -d "${ENV_PREFIX}" ]]; then
  conda create -y -p "${ENV_PREFIX}" python=3.10
fi
conda activate "${ENV_PREFIX}"

export PIP_CACHE_DIR="${SCRATCH_BASE}/pip_cache"
export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache"
export TMPDIR="${SCRATCH_BASE}/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}"

# ---- Run config ----
export BASE_MODEL="${BASE_MODEL:-google/gemma-2-2b-it}"
export OPENAI_HINT_MODEL="${OPENAI_HINT_MODEL:-gpt-5-mini}"

python - <<'PY'
import os, subprocess, sys
from importlib.metadata import version as pkg_version

def pip_install(args):
    # IMPORTANT: Don't force-reinstall on every run; it causes multi-GB downloads and very slow startup.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *args])

def pip_uninstall(pkgs):
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", *pkgs])

torch_index = os.environ.get("TORCH_INDEX_URL")
torch_extra = ["--index-url", torch_index] if torch_index else []

# Optional: set FORCE_REINSTALL=1 to refresh installs.
force = os.environ.get("FORCE_REINSTALL", "0") == "1"
if force:
    pip_uninstall(["bitsandbytes", "triton", "fsspec", "torch", "transformers", "accelerate", "peft", "trl", "datasets", "openai"])

# torch first (CUDA wheel) — only install if missing or forcing.
try:
    import torch  # noqa: F401
except Exception:
    pip_install([*torch_extra, "torch"])

# HF + RLFT stack — install if missing or wrong.
tf_ver = "4.51.0"

reqs = [
    f"transformers=={tf_ver}",
    "accelerate==0.33.0",
    "peft==0.12.0",
    "trl==0.10.1",
    "datasets==2.20.0",
    "fsspec[http]==2024.5.0",
    "rich==13.7.1",
    "sentencepiece",
    "bitsandbytes>=0.43.0",  # For 4-bit quantization
]
need = force
if not need:
    try:
        ok = (
            pkg_version("transformers") == tf_ver
            and pkg_version("accelerate") == "0.33.0"
            and pkg_version("peft") == "0.12.0"
            and pkg_version("trl") == "0.10.1"
            and pkg_version("datasets") == "2.20.0"
            and pkg_version("fsspec") == "2024.5.0"
            and pkg_version("rich") == "13.7.1"
            and pkg_version("bitsandbytes") is not None
        )
        need = not ok
    except Exception:
        need = True
if need:
    pip_install(reqs)

# OpenAI client for hint generation (install if missing)
try:
    import openai  # noqa: F401
except Exception:
    pip_install(["openai>=1.40.0"])

print("[versions] transformers", pkg_version("transformers"))
print("[versions] accelerate", pkg_version("accelerate"))
print("[versions] bitsandbytes", pkg_version("bitsandbytes"))
print("[versions] peft", pkg_version("peft"))
print("[versions] trl", pkg_version("trl"))
print("[versions] datasets", pkg_version("datasets"))
print("[versions] fsspec", pkg_version("fsspec"))
print("[versions] rich", pkg_version("rich"))
print("[versions] openai", pkg_version("openai"))
print("[config] BASE_MODEL", os.environ.get("BASE_MODEL"))
PY

echo "[env] Python: $(python --version)"
echo "[env] Node: $(hostname)"
echo "[env] Date: $(date)"
echo "[env] GPU:"
nvidia-smi || true

RUNS_ROOT="${HOME}/scratch/mst_gsm_poc_runs"
mkdir -p "${RUNS_ROOT}"

# Optional: hardcoded fallback dataset dir (known-good).
# Leave empty by default (recommended for sharing publicly).
DATASET_DIR_FALLBACK="${DATASET_DIR_FALLBACK:-}"

# Find the newest *existing* dataset dir BEFORE creating OUTDIR (so we don't select the new run dir).
DATASET_DIR=""
for d in $(ls -1dt "${RUNS_ROOT}"/* 2>/dev/null || true); do
  if [[ -f "${d}/train_dataset.jsonl" ]] && [[ -f "${d}/eval_dataset.jsonl" ]] && [[ -f "${d}/hint_cache.jsonl" ]]; then
    DATASET_DIR="${d}"
    break
  fi
done

if [[ -z "${DATASET_DIR}" ]] && [[ -n "${DATASET_DIR_FALLBACK}" ]] && [[ -d "${DATASET_DIR_FALLBACK}" ]]; then
  DATASET_DIR="${DATASET_DIR_FALLBACK}"
fi

# Always write *new* training outputs to a fresh timestamped directory.
OUTDIR="${RUNS_ROOT}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTDIR}"

if [[ -z "${DATASET_DIR}" ]]; then
  DATASET_DIR="${OUTDIR}"
fi

TRAIN_JSONL="${DATASET_DIR}/train_dataset.jsonl"
VAL_JSONL="${DATASET_DIR}/val_dataset.jsonl"
EVAL_JSONL="${DATASET_DIR}/eval_dataset.jsonl"
HINT_CACHE="${DATASET_DIR}/hint_cache.jsonl"

echo "[startup] OUTDIR=${OUTDIR}" >&2
echo "[startup] DATASET_DIR=${DATASET_DIR}" >&2
echo "[startup] FORCE_REINSTALL=${FORCE_REINSTALL:-0}" >&2

# ---- Project directory (PACE) ----
# You do NOT need a full repo; it's enough to have a folder containing:
#   - run_mst_gsm_poc.slurm
#   - mst_gsm_poc.py
#   - build_gsm_hint_dataset.py
#   - gsm_hint_dataset.py
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
echo "[startup] PROJECT_DIR=${PROJECT_DIR}" >&2
cd "${PROJECT_DIR}"

# Fail fast if this job was staged without the full folder (common if someone does `sbatch mst_gsm_poc.py`).
if [[ ! -f "${PROJECT_DIR}/gsm_hint_dataset.py" ]]; then
  echo "[error] Missing ${PROJECT_DIR}/gsm_hint_dataset.py" >&2
  echo "[hint] Submit the SLURM script from the folder that contains all 4 files, e.g.:" >&2
  echo "       sbatch \"${PROJECT_DIR}/run_mst_gsm_poc.slurm\"" >&2
  echo "[hint] Or, if submitting from elsewhere:" >&2
  echo "       sbatch --export=ALL,PROJECT_DIR=\"/home/hice1/aharris345/mst_gsm_poc\" \"/home/hice1/aharris345/mst_gsm_poc/run_mst_gsm_poc.slurm\"" >&2
  exit 2
fi

# Support two common layouts:
# - PROJECT_DIR is repo root:           ${PROJECT_DIR}/GSM/mst_gsm_poc.py
# - PROJECT_DIR is already the GSM dir: ${PROJECT_DIR}/mst_gsm_poc.py
SCRIPT=""
if [[ -f "${PROJECT_DIR}/GSM/mst_gsm_poc.py" ]]; then
  SCRIPT="${PROJECT_DIR}/GSM/mst_gsm_poc.py"
elif [[ -f "${PROJECT_DIR}/mst_gsm_poc.py" ]]; then
  SCRIPT="${PROJECT_DIR}/mst_gsm_poc.py"
else
  echo "[error] Could not find GSM MST script under PROJECT_DIR=${PROJECT_DIR}" >&2
  echo "[tried] ${PROJECT_DIR}/GSM/mst_gsm_poc.py" >&2
  echo "[tried] ${PROJECT_DIR}/mst_gsm_poc.py" >&2
  exit 2
fi

# ---- Build dataset first (so you can inspect it while training runs) ----
DATASET_SCRIPT=""
if [[ -f "${PROJECT_DIR}/GSM/build_gsm_hint_dataset.py" ]]; then
  DATASET_SCRIPT="${PROJECT_DIR}/GSM/build_gsm_hint_dataset.py"
elif [[ -f "${PROJECT_DIR}/build_gsm_hint_dataset.py" ]]; then
  DATASET_SCRIPT="${PROJECT_DIR}/build_gsm_hint_dataset.py"
else
  echo "[error] Could not find dataset builder under PROJECT_DIR=${PROJECT_DIR}" >&2
  echo "[tried] ${PROJECT_DIR}/GSM/build_gsm_hint_dataset.py" >&2
  echo "[tried] ${PROJECT_DIR}/build_gsm_hint_dataset.py" >&2
  exit 2
fi

if [[ -f "${TRAIN_JSONL}" ]] && [[ -f "${EVAL_JSONL}" ]] && [[ -f "${HINT_CACHE}" ]]; then
  echo "[dataset] reusing latest dataset dir: ${DATASET_DIR}" >&2
else
  # If the latest dir doesn't have a dataset yet, build the dataset into OUTDIR and reuse that.
  DATASET_DIR="${OUTDIR}"
  TRAIN_JSONL="${DATASET_DIR}/train_dataset.jsonl"
  VAL_JSONL="${DATASET_DIR}/val_dataset.jsonl"
  EVAL_JSONL="${DATASET_DIR}/eval_dataset.jsonl"
  HINT_CACHE="${DATASET_DIR}/hint_cache.jsonl"

  echo "[dataset] no reusable dataset found; building dataset into: ${DATASET_DIR}" >&2
  python -u "${DATASET_SCRIPT}" \
    --out_dir "${DATASET_DIR}" \
    --openai_hint_model "${OPENAI_HINT_MODEL}" \
    --hint_cache_path "${HINT_CACHE}" \
    --seed 7 \
    --n_train_tasks 2560 \
    --n_val_tasks 640 \
    --n_eval_tasks 1280 \
    --user_answer_correct_frac 0.5 \
    --preview_n 8
fi

echo "[dataset] TRAIN_JSONL=${TRAIN_JSONL}" >&2
echo "[dataset] VAL_JSONL=${VAL_JSONL}" >&2
echo "[dataset] EVAL_JSONL=${EVAL_JSONL}" >&2
echo "[dataset] HINT_CACHE=${HINT_CACHE}" >&2

python -u "${SCRIPT}" \
  --base_model "${BASE_MODEL}" \
  --openai_hint_model "${OPENAI_HINT_MODEL}" \
  --output_dir "${OUTDIR}" \
  --hint_cache_path "${HINT_CACHE}" \
  --train_dataset_jsonl "${TRAIN_JSONL}" \
  --val_dataset_jsonl "${VAL_JSONL}" \
  --eval_dataset_jsonl "${EVAL_JSONL}" \
  --dataset_dir "${DATASET_DIR}" \
  --preview_n 8 \
  --run_all \
  --skip_test_eval \
  --trust_remote_code \
  --device_map "cuda:0" \
  --steps 200 \
  --batch_size 128 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --n_train_tasks 2560 \
  --n_val_tasks 640 \
  --n_eval_tasks 1280 \
  --user_answer_correct_frac 0.5 \
  --p_min 0.25 \
  --p_max 1.0 \
  --gen_max_new_tokens 16 \
  --gen_temperature 1.6

echo "[done] Outputs in ${OUTDIR}"


