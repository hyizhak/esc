#!/usr/bin/env python3

"""
Utility script to launch a grid search over the GNN hyper-parameters used by run.py.

Each hyper-parameter combination gets its own model directory under --model_root.
"""

import argparse
import csv
import itertools
import os
import queue
import shlex
import subprocess
import sys
from threading import Lock, Thread
from typing import Dict, List, Optional, Tuple


MODEL_FILENAME = "model.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search launcher for run.py GNN hyper-parameters")
    parser.add_argument("--run_py", default="run.py", help="Path to run.py (default: run.py in current dir)")
    parser.add_argument("--exp_dir", required=True,
                        help="Base experiment directory (expects dev-text/dev-m2/test-text/test-m2 layout)")
    parser.add_argument("--session", required=True,
                        help="Session name used to group models/outputs/results")
    parser.add_argument("--train_data_dir", help="Override path to training data directory (default: EXP_DIR/dev-text)")
    parser.add_argument("--train_m2_dir", help="Override path to training m2 directory (default: EXP_DIR/dev-m2)")
    parser.add_argument("--test_data_dir", help="Override path to test data directory (default: EXP_DIR/test-text)")
    parser.add_argument("--test_m2_dir", help="Override path to test m2 directory (default: EXP_DIR/test-m2)")
    parser.add_argument("--source_name", default="source.txt", help="Source filename (matches original command)")
    parser.add_argument("--target_name", default="target.txt", help="Target filename (matches original command)")
    parser.add_argument("--max_hypotheses", type=int, nargs="+", default=[None],
                        help="Maximum number of hypothesis files to include in run.py")
    parser.add_argument("--model_root", help="Directory to store model subfolders (default: EXP_DIR/models)")
    parser.add_argument("--vocab_template", default="{model_dir}/vocab.idx",
                        help="Template for vocab path. Use {model_dir} placeholder. Default: per-run vocab inside model dir.")
    parser.add_argument("--outputs_root", help="Base directory for test outputs (default: EXP_DIR/outputs)")
    parser.add_argument("--output_template", default="{run_id}.out",
                        help="Filename template (relative to outputs_root) for test predictions.")
    parser.add_argument("--gold_m2", help="Path to gold .m2 file for scoring (required when evaluation enabled)")
    parser.add_argument("--m2_scorer_py", default="m2scorer/scripts/m2scorer.py",
                        help="Path to the m2scorer Python script")
    parser.add_argument("--scorer_python", default=None,
                        help="Python executable to run the m2 scorer (default: same as this script)")
    parser.add_argument("--results_csv", help="CSV file to write aggregated metrics "
                                              "(default: <model_root>/grid_results.csv)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Train models only; skip test generation and scoring")
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)),
                        help="List of GPU IDs to cycle through (default: 0-7)")
    parser.add_argument("--val_ratio", type=int, default=5, help="Validation split ratio passed to run.py")
    parser.add_argument("--weight_decay_list", type=float, nargs="+", default=[0.01],
                        help="List of weight decay values")
    parser.add_argument("--lr_list", type=float, nargs="+", default=[0.005],
                        help="List of learning rates")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64],
                        help="List of GNN hidden sizes")
    parser.add_argument("--layers", type=int, nargs="+", default=[3],
                        help="List of GraphSAGE layer counts")
    parser.add_argument("--dropouts", type=float, nargs="+", default=[0.1],
                        help="List of dropout rates between GNN layers")
    parser.add_argument("--edit_type_dims", type=int, nargs="+", default=[64],
                        help="List of edit-type encoder dimensions")
    parser.add_argument("--hypothesis_dims", type=int, nargs="+", default=[32],
                        help="List of hypothesis encoder dimensions")
    parser.add_argument("--scalar_feat_dims", type=int, nargs="+", default=[32],
                        help="List of scalar feature encoder dimensions")
    parser.add_argument("--ff_multipliers", type=float, nargs="+", default=[2.0],
                        help="List of feed-forward expansion ratios within each GNN block")
    parser.add_argument("--context_layers_list", type=int, nargs="+", default=[1],
                        help="List of context layer counts applied after the GNN stack")
    parser.add_argument("--context_heads_list", type=int, nargs="+", default=[4],
                        help="List of attention head counts for the context layers")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="Random seeds to try")
    parser.add_argument("--priority_loss_weights", type=float, nargs="+", default=[0.8],
                        help="List of priority loss weights to try")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Early stopping patience passed to run.py (0 disables)")
    parser.add_argument("--upsample", type=str, default=None,
                        help="Optional up-sampling ratio string passed to run.py (e.g., '1:2')")
    parser.add_argument("--beam_sizes", type=int, nargs="+", default=[1],
                        help="Beam sizes to use during decoding (1 disables beam search)")
    parser.add_argument("--beam_priority_weights", type=float, nargs="+", default=[0.5],
                        help="Priority score weights used inside beam decoding")
    parser.add_argument("--beam_min_probs", type=float, nargs="+", default=[0.2],
                        help="Minimum keep probability for edits entering the beam search")
    parser.add_argument("--errant_parallel_cmd", default="errant_parallel",
                        help="Command to run errant_parallel when using BEA pipeline")
    parser.add_argument("--errant_compare_cmd", default="errant_compare",
                        help="Command to run errant_compare when using BEA pipeline")
    parser.add_argument("--errant_source_path",
                        help="Path passed to errant_parallel -ori (default: <train_data_dir>/<source_name>)")
    parser.add_argument("--extra_args", type=str, default="",
                        help="Additional arguments appended verbatim (e.g., \"--threshold 0.6\")")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the commands without executing them")
    return parser.parse_args()


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def build_command(python_bin: str,
                  run_py: str,
                  args,
                  hidden_dim: int,
                  layers: int,
                  dropout: float,
                  lr: float,
                  weight_decay: float,
                  seed: int,
                  priority_loss_weight: float,
                  edit_type_dim: int,
                  hypothesis_dim: int,
                  scalar_feat_dim: int,
                  ff_multiplier: float,
                  context_layers: int,
                  context_heads: int,
                  max_hypotheses: Optional[int],
                  model_dir: str,
                  vocab_path: str,
                  data_dir: str,
                  m2_dir: str) -> List[str]:
    cmd = [
        python_bin,
        run_py,
        "--data_dir", data_dir,
        "--m2_dir", m2_dir,
        "--source_name", args.source_name,
        "--target_name", args.target_name,
        "--lr", str(lr),
        "--weight_decay", str(weight_decay),
        "--seed", str(seed),
        "--val_ratio", str(args.val_ratio),
        "--gnn_hidden_dim", str(hidden_dim),
        "--gnn_layers", str(layers),
        "--gnn_dropout", str(dropout),
        "--edit_type_dim", str(edit_type_dim),
        "--hypothesis_dim", str(hypothesis_dim),
        "--scalar_feat_dim", str(scalar_feat_dim),
        "--ff_multiplier", str(ff_multiplier),
        "--context_layers", str(context_layers),
        "--context_heads", str(context_heads),
        "--max_hypotheses", str(max_hypotheses) if max_hypotheses is not None else "0",
        "--vocab_path", vocab_path,
        "--model_path", model_dir,
        "--train",
        "--priority_loss_weight", str(priority_loss_weight),
        "--early_stop_patience", str(args.early_stop_patience),
    ]
    if args.upsample:
        cmd.extend(["--upsample", args.upsample])
    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def execute(cmd: List[str], dry_run: bool = False, capture_output: bool = False,
            env: Optional[Dict[str, str]] = None) -> Tuple[bool, Optional[str]]:
    print("Running:", " ".join(cmd))
    if dry_run:
        return True, ""
    try:
        if capture_output:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                env=env,
            )
            return True, result.stdout
        else:
            subprocess.run(cmd, check=True, env=env)
            return True, ""
    except subprocess.CalledProcessError as exc:
        if capture_output:
            sys.stderr.write(exc.stdout or "")
            sys.stderr.write(exc.stderr or "")
        else:
            sys.stderr.write(exc.stderr or "")
        sys.stderr.write(f"Command failed with exit code {exc.returncode}\n")
        return False, None


def parse_m2_output(output: str) -> dict:
    metrics = {}
    if not output:
        return metrics
    lines = [line.strip() for line in output.splitlines()]
    for line in lines:
        if line.startswith("Precision"):
            metrics["precision"] = float(line.split(":")[1].strip())
        elif line.startswith("Recall"):
            metrics["recall"] = float(line.split(":")[1].strip())
        elif line.startswith("F_"):
            metrics["f05"] = float(line.split(":")[1].strip())
    if {"precision", "recall", "f05"}.issubset(metrics.keys()):
        return metrics

    header_keys = {"TP", "FP", "FN", "Prec", "Rec", "F0.5"}
    for idx, line in enumerate(lines):
        tokens = line.split()
        if not tokens:
            continue
        if header_keys.issubset(set(tokens)) and idx + 1 < len(lines):
            data_tokens = lines[idx + 1].split()
            header_to_idx = {token: i for i, token in enumerate(tokens)}
            try:
                metrics["precision"] = float(data_tokens[header_to_idx["Prec"]])
                metrics["recall"] = float(data_tokens[header_to_idx["Rec"]])
                metrics["f05"] = float(data_tokens[header_to_idx["F0.5"]])
                return metrics
            except (KeyError, ValueError, IndexError):
                continue
    return metrics


def process_run(job: dict,
                gpu_id: Optional[int],
                args,
                python_bin: str,
                scorer_python: str,
                paths: dict,
                dry_run: bool = False) -> Optional[dict]:
    run_id = job["run_id"]
    idx = job["idx"]
    total = job["total"]
    gpu_label = f"GPU{gpu_id}" if gpu_id is not None else "CPU"
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model_dir = os.path.join(paths["model_root"], run_id)
    vocab_path = args.vocab_template.format(model_dir=model_dir, run_id=run_id)
    ensure_dir(model_dir)
    vocab_dir = os.path.dirname(vocab_path)
    if vocab_dir:
        ensure_dir(vocab_dir)

    checkpoint_path = os.path.join(model_dir, MODEL_FILENAME)
    if not dry_run and os.path.isfile(checkpoint_path):
        print(f"[{idx}/{total}] [{gpu_label}] Skipping {run_id}, {MODEL_FILENAME} already exists.")
        
    else:

        train_cmd = build_command(
            python_bin,
            args.run_py,
            args,
            job["hidden_dim"],
            job["layers"],
            job["dropout"],
            job["lr"],
            job["weight_decay"],
            job["seed"],
            job["priority_loss_weight"],
            job["edit_type_dim"],
            job["hypothesis_dim"],
            job["scalar_feat_dim"],
            job["ff_multiplier"],
            job["context_layers"],
            job["context_heads"],
            job["max_hypotheses"],
            model_dir,
            vocab_path,
            paths["train_data_dir"],
            paths["train_m2_dir"],
        )

        print(f"[{idx}/{total}] [{gpu_label}] Training {run_id}")
        success, _ = execute(train_cmd, dry_run=dry_run, env=env)
        if not success:
            return None

        if dry_run or args.skip_eval:
            return None

    model_pt = checkpoint_path
    output_filename = args.output_template.format(run_id=run_id, model_dir=model_dir)
    output_path = os.path.join(paths["outputs_root"], output_filename)
    ensure_dir(os.path.dirname(output_path))

    test_cmd = [
        python_bin,
        args.run_py,
        "--test",
        "--data_dir", paths["test_data_dir"],
        "--m2_dir", paths["test_m2_dir"],
        "--source_name", args.source_name,
        "--target_name", args.target_name,
        "--gnn_hidden_dim", str(job["hidden_dim"]),
        "--gnn_layers", str(job["layers"]),
        "--gnn_dropout", str(job["dropout"]),
        "--edit_type_dim", str(job["edit_type_dim"]),
        "--hypothesis_dim", str(job["hypothesis_dim"]),
        "--scalar_feat_dim", str(job["scalar_feat_dim"]),
        "--ff_multiplier", str(job["ff_multiplier"]),
        "--context_layers", str(job["context_layers"]),
        "--context_heads", str(job["context_heads"]),
        "--beam_size", str(job["beam_size"]),
        "--beam_priority_weight", str(job["beam_priority_weight"]),
        "--beam_min_prob", str(job["beam_min_prob"]),
        "--model_path", model_pt,
        "--vocab_path", vocab_path,
        "--output_path", output_path,
    ]
    if args.max_hypotheses is not None:
        test_cmd.extend(["--max_hypotheses", str(job["max_hypotheses"])])
    print(f"[{idx}/{total}] [{gpu_label}] Testing {run_id}")
    success, _ = execute(test_cmd, env=env)
    if not success:
        return None

    exp_name = os.path.basename(os.path.normpath(args.exp_dir)).lower()
    dataset_type = "bea" if "bea" in exp_name else "conll"
    metrics = {}
    extra_fields = {}
    if dataset_type == "bea":
        hypo_m2_path = output_path + ".m2"
        ori_path = args.errant_source_path or os.path.join(paths["train_data_dir"], args.source_name)
        errant_parallel_cmd = [
            args.errant_parallel_cmd,
            "-ori", ori_path,
            "-cor", output_path,
            "-out", hypo_m2_path,
        ]
        print(f"[{idx}/{total}] [{gpu_label}] Converting predictions to M2 for {run_id}")
        success, _ = execute(errant_parallel_cmd, env=env)
        if not success:
            return None
        errant_compare_cmd = [
            args.errant_compare_cmd,
            "-ref", args.gold_m2,
            "-hyp", hypo_m2_path,
        ]
        print(f"[{idx}/{total}] [{gpu_label}] Scoring {run_id} with ERRANT")
        success, scorer_output = execute(errant_compare_cmd, capture_output=True)
        metrics = parse_m2_output(scorer_output if success else "")
        extra_fields["hyp_m2_path"] = hypo_m2_path
    else:
        score_cmd = [
            scorer_python,
            args.m2_scorer_py,
            output_path,
            args.gold_m2,
        ]
        print(f"[{idx}/{total}] [{gpu_label}] Scoring {run_id}")
        success, scorer_output = execute(score_cmd, capture_output=True)
        metrics = parse_m2_output(scorer_output if success else "")
        extra_fields["hyp_m2_path"] = None

    row = {
        "run_id": run_id,
        "hidden_dim": job["hidden_dim"],
        "layers": job["layers"],
        "dropout": job["dropout"],
        "lr": job["lr"],
        "weight_decay": job["weight_decay"],
        "seed": job["seed"],
        "priority_loss_weight": job["priority_loss_weight"],
        "edit_type_dim": job["edit_type_dim"],
        "hypothesis_dim": job["hypothesis_dim"],
        "scalar_feat_dim": job["scalar_feat_dim"],
        "ff_multiplier": job["ff_multiplier"],
        "context_layers": job["context_layers"],
        "context_heads": job["context_heads"],
        "beam_size": job["beam_size"],
        "beam_priority_weight": job["beam_priority_weight"],
        "beam_min_prob": job["beam_min_prob"],
        "max_hypotheses": job["max_hypotheses"],
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f0.5": metrics.get("f05"),
        "dataset_type": dataset_type,
        "model_dir": model_dir,
        "output_path": output_path,
        "model_path": model_pt,
        "hyp_m2_path": extra_fields["hyp_m2_path"],
        "gpu": gpu_id,
    }
    return row


def main():
    args = parse_args()
    python_bin = sys.executable
    scorer_python = args.scorer_python or python_bin
    train_data_dir = args.train_data_dir or os.path.join(args.exp_dir, "dev-text")
    train_m2_dir = args.train_m2_dir or os.path.join(args.exp_dir, "dev-m2")
    test_data_dir = args.test_data_dir or os.path.join(args.exp_dir, "test-text") if args.exp_dir == "conll-exp" else os.path.join(args.exp_dir, "dev-text")
    test_m2_dir = args.test_m2_dir or os.path.join(args.exp_dir, "test-m2") if args.exp_dir == "conll-exp" else os.path.join(args.exp_dir, "dev-m2")
    model_root = args.model_root or os.path.join(args.exp_dir, "models", args.session)
    outputs_root = args.outputs_root or os.path.join(args.exp_dir, "outputs", args.session)
    results_csv = args.results_csv or os.path.join(model_root, "grid_results.csv")
    ensure_dir(model_root)
    ensure_dir(outputs_root)

    if not args.skip_eval and not args.gold_m2:
        raise ValueError("--gold_m2 must be provided unless --skip_eval is set.")

    def fmt(val):
        if isinstance(val, float):
            if float(val).is_integer():
                return str(int(val))
            return str(val).replace('.', 'p')
        return str(val)

    combos = list(itertools.product(
        args.hidden_dims,
        args.layers,
        args.dropouts,
        args.lr_list,
        args.weight_decay_list,
        args.seeds,
        args.priority_loss_weights,
        args.edit_type_dims,
        args.hypothesis_dims,
        args.scalar_feat_dims,
        args.ff_multipliers,
        args.context_layers_list,
        args.context_heads_list,
        args.beam_sizes,
        args.beam_priority_weights,
        args.beam_min_probs,
        args.max_hypotheses,
    ))

    print("Planned runs: {}".format(len(combos)))
    jobs = []
    total = len(combos)
    for idx, combo in enumerate(combos, 1):
        (hidden_dim, layers, dropout, lr, weight_decay, seed, p_loss_w,
         edit_type_dim, hypothesis_dim, scalar_feat_dim, ff_multiplier,
         context_layers, context_heads, beam_size, beam_priority_weight,
         beam_min_prob, max_hypotheses) = combo
        run_id = (
            f"hd{hidden_dim}_L{layers}_drop{fmt(dropout)}_lr{fmt(lr)}_wd{fmt(weight_decay)}"
            f"_seed{seed}_plw{fmt(p_loss_w)}_edit{edit_type_dim}_hyp{hypothesis_dim}"
            f"_scalar{scalar_feat_dim}_ff{fmt(ff_multiplier)}"
            f"_ctxL{context_layers}_ctxH{context_heads}"
            f"_beam{beam_size}_bpw{fmt(beam_priority_weight)}_bmp{fmt(beam_min_prob)}_nh{fmt(max_hypotheses)}"
        )
        jobs.append({
            "idx": idx,
            "total": total,
            "run_id": run_id,
            "hidden_dim": hidden_dim,
            "layers": layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
            "priority_loss_weight": p_loss_w,
            "edit_type_dim": edit_type_dim,
            "hypothesis_dim": hypothesis_dim,
            "scalar_feat_dim": scalar_feat_dim,
            "ff_multiplier": ff_multiplier,
            "context_layers": context_layers,
            "context_heads": context_heads,
            "beam_size": beam_size,
            "beam_priority_weight": beam_priority_weight,
            "beam_min_prob": beam_min_prob,
            "max_hypotheses": max_hypotheses,
        })

    paths = {
        "train_data_dir": train_data_dir,
        "train_m2_dir": train_m2_dir,
        "test_data_dir": test_data_dir,
        "test_m2_dir": test_m2_dir,
        "model_root": model_root,
        "outputs_root": outputs_root,
    }

    aggregated_rows = []
    if args.dry_run:
        for job in jobs:
            process_run(job, None, args, python_bin, scorer_python, paths, dry_run=True)
    else:
        combo_queue = queue.Queue()
        for job in jobs:
            combo_queue.put(job)

        lock = Lock()

        def worker(gpu_id):
            while True:
                try:
                    job = combo_queue.get_nowait()
                except queue.Empty:
                    break
                row = process_run(job, gpu_id, args, python_bin, scorer_python, paths, dry_run=False)
                if row:
                    with lock:
                        aggregated_rows.append(row)

        threads = []
        for gpu_id in args.gpus:
            t = Thread(target=worker, args=(gpu_id,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    if not args.dry_run and aggregated_rows:
        fieldnames = ["run_id", "hidden_dim", "layers", "dropout", "lr",
                      "weight_decay", "seed", "priority_loss_weight",
                      "edit_type_dim", "hypothesis_dim", "scalar_feat_dim",
                      "ff_multiplier", "context_layers", "context_heads",
                      "beam_size", "beam_priority_weight", "beam_min_prob",
                      "max_hypotheses", "hypotheses",
                      "precision", "recall", "f0.5", "dataset_type",
                      "model_dir", "output_path", "model_path", "hyp_m2_path", "gpu"]
        print(f"Writing aggregated metrics to {results_csv}")
        with open(results_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregated_rows)


if __name__ == "__main__":
    main()
