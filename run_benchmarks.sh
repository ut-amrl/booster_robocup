#!/usr/bin/env bash

# Lightweight helper to run the test script with standardized options on
# each test subtask. This forwards args to `scripts/rsl_rl/test.py`
# after some minimal validation and exposes a nicer shell-level CLI.

set -euo pipefail

SUBTASKS=("walk" "run" "uneven" "push")

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$ROOT_DIR/scripts/rsl_rl/test.py"

print_help() {
	cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --checkpoint PATH             Use local checkpoint file or path
  --wandb                       Load model from wandb instead (uses --wandb_run / --wandb_model or will prompt)
  --wandb_run RUN_PATH          WandB run path
  --wandb_model MODEL           WandB model name
  --wandb_log_run RUN_PATH      WandB run path to log benchmark results to
  -h, --help                    Show this help

Examples:
  $(basename "$0") --checkpoint path/to/checkpoint --wandb_log_run user/project/abcd1234
  $(basename "$0") --wandb --wandb_run path/to/run --wandb_model model_30099.pt

EOF
}

if [[ ! -f "$SCRIPT_PATH" ]]; then
	echo "[ERROR] Cannot find test script at: $SCRIPT_PATH" >&2
	exit 2
fi

# default values
checkpoint=""
use_wandb="false"
wandb_run=""
wandb_model=""
wandb_log_run=""

POSITIONAL=()
while [[ $# -gt 0 ]]; do
	key="$1"
	case $key in
		--checkpoint)
		checkpoint="$2"; shift; shift;;
		--wandb)
		use_wandb="true"; shift;;
		--wandb_run)
		wandb_run="$2"; shift; shift;;
		--wandb_model)
		wandb_model="$2"; shift; shift;;
		--wandb_log_run)
		wandb_log_run="$2"; shift; shift;;
		-h|--help)
		print_help; exit 0;;
		--) shift; break;;
		-*|--*)
		echo "Unknown option $1"; print_help; exit 1;;
		*)
		POSITIONAL+=("$1"); shift;;
	esac
done

if [[ -z "$checkpoint" && "$use_wandb" != "true" ]]; then
    echo "[ERROR] Must specify either --checkpoint or --wandb" >&2
    print_help
    exit 1
fi

run_cmd=("$ROOT_DIR/isaaclab.sh" -p "$SCRIPT_PATH")
args=(
  --task "T1-Baseline-Benchmark-v0" \
  --num_envs 64 \
  --max_length 2048 \
  --video \
  --headless \
)

if [[ -n "$checkpoint" ]]; then
	args+=(--checkpoint "$checkpoint")
else
	args+=(--wandb --wandb_run "$wandb_run" --wandb_model "$wandb_model")
fi

[[ -n "$wandb_log_run" ]] && cmd+=(--wandb_log_run "$wandb_log_run")

for subtask in "${SUBTASKS[@]}"; do
    echo "[INFO] Running benchmark for subtask: $subtask"
    echo "      Command: ${run_cmd[*]} --subtask $subtask ${args[*]}"
    "${run_cmd[@]}" --subtask "$subtask" "${args[@]}" 
done