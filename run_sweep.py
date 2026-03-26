#!/usr/bin/env python3
"""
AIConfigurator Sweep -> CSV
============================
Usage:
    source venv-aic/bin/activate

    # Everything: all models x all GPUs x all backends (~4 hours)
    python run_sweep.py --full

    # Specific model + GPU (auto TP/BS enumeration, ~1 min)
    python run_sweep.py --model Qwen/Qwen3-32B-FP8 --system h200_sxm

    # Specific model + GPU + backend
    python run_sweep.py --model Qwen/Qwen3-32B-FP8 --system h200_sxm --backend vllm
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


WORKLOADS = [
    (128,   32),
    (256,  128),
    (512,  128),
    (512,  512),
    (1024, 256),
    (1024, 1024),
    (2048, 512),
    (2048, 2048),
    (4096, 512),
    (4096, 1000),
    (4096, 2048),
    (8192, 1024),
    (8192, 2048),
    (16384, 512),
    (16384, 2048),
]

TP_LIST = [1, 2, 4, 8]
PP_LIST = [1, 2, 3, 4]
BS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def load_support_triples():
    for p in [
        Path(__file__).parent / "src" / "aiconfigurator" / "systems" / "support_matrix.csv",
        Path("src/aiconfigurator/systems/support_matrix.csv"),
    ]:
        if p.exists():
            sm = pd.read_csv(p)
            passed = sm[sm["Status"] == "PASS"]
            triples = passed[["HuggingFaceID", "System", "Backend"]].drop_duplicates()
            return [(r["HuggingFaceID"], r["System"], r["Backend"]) for _, r in triples.iterrows()]
    sys.exit("Error: support_matrix.csv not found")


# Models to skip in full sweep
SKIP_MODELS = {"Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"}


def probe_max_bs(model, system, backend, tp, pp):
    """Find max valid BS for this (TP, PP) using worst-case workload."""
    from aiconfigurator.cli.api import cli_estimate
    worst_isl = max(w[0] for w in WORKLOADS)
    worst_osl = max(w[1] for w in WORKLOADS)
    max_bs = 0
    for bs in BS_LIST:
        try:
            cli_estimate(model_path=model, system_name=system, backend_name=backend,
                         tp_size=tp, pp_size=pp, batch_size=bs, isl=worst_isl, osl=worst_osl)
            max_bs = bs
        except Exception:
            break
    return max_bs


def sweep_one_triple(model, system, backend, label=""):
    """Run full sweep for one (model, system, backend). Returns list of row dicts."""
    from aiconfigurator.cli.api import cli_estimate

    # Probe (TP, PP) / BS limits
    tp_pp_max = {}
    for tp in TP_LIST:
        for pp in PP_LIST:
            mb = probe_max_bs(model, system, backend, tp, pp)
            if mb > 0:
                tp_pp_max[(tp, pp)] = mb
                print(f"  {label}TP={tp} PP={pp}: max BS={mb}")
            else:
                print(f"  {label}TP={tp} PP={pp}: doesn't fit")

    if not tp_pp_max:
        return []

    # Build all combos
    combos = []
    for (tp, pp), max_bs in tp_pp_max.items():
        for bs in BS_LIST:
            if bs > max_bs:
                continue
            for isl, osl in WORKLOADS:
                combos.append((tp, pp, bs, isl, osl))

    print(f"  {label}Running {len(combos)} configs...")
    rows = []
    failed = 0
    t0 = time.time()

    for idx, (tp, pp, bs, isl, osl) in enumerate(combos):
        try:
            r = cli_estimate(model_path=model, system_name=system, backend_name=backend,
                             tp_size=tp, pp_size=pp, batch_size=bs, isl=isl, osl=osl)
            row = dict(r.raw)
            if r.per_ops_data:
                for phase, ops in r.per_ops_data.items():
                    if isinstance(ops, dict):
                        for op_name, val in ops.items():
                            row[f"perop_{phase}_{op_name}"] = val
            rows.append(row)
        except Exception:
            failed += 1

        done = idx + 1
        if done % 100 == 0 or done == len(combos):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(combos) - done) / rate if rate > 0 else 0
            print(f"\r  {label}[{done}/{len(combos)}] {len(rows)} ok, {failed} fail ({rate:.1f}/s, ETA {eta:.0f}s)   ",
                  end="", flush=True)
    print()
    return rows


def save_csv(rows, output):
    if not rows:
        print("No valid results to save.")
        return
    df = pd.DataFrame(rows)
    priority = [
        "model", "system", "backend", "version",
        "tp", "pp", "dp", "moe_tp", "moe_ep", "parallel",
        "bs", "isl", "osl",
        "ttft", "tpot", "request_latency",
        "tokens/s", "tokens/s/gpu", "tokens/s/user", "seq/s", "seq/s/gpu",
        "concurrency", "memory",
        "gemm", "kvcache", "fmha", "moe", "comm", "power_w",
    ]
    front = [c for c in priority if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]
    df.to_csv(output, index=False)
    print(f"Saved: {output}  ({len(df)} rows x {len(df.columns)} columns)")


def main():
    parser = argparse.ArgumentParser(description="AIConfigurator sweep -> CSV")
    parser.add_argument("--full", action="store_true",
                        help="Run ALL models x ALL GPUs x ALL backends. Saves full_sweep.csv")
    parser.add_argument("--model",   default=None, help="HuggingFace model ID")
    parser.add_argument("--system",  default=None, help="GPU system (e.g. h200_sxm)")
    parser.add_argument("--backend", default=None, help="Backend (default: sweep all supported)")
    parser.add_argument("--output",  default=None, help="Output CSV (auto-named if omitted)")
    args = parser.parse_args()

    t0 = time.time()

    if args.full:
        # ── Full: all supported triples ──
        triples = load_support_triples()

        # Remove skipped models
        triples = [(m, s, b) for m, s, b in triples if m not in SKIP_MODELS]

        # Filter by --backend if specified
        if args.backend:
            triples = [(m, s, b) for m, s, b in triples if b == args.backend]
            print(f"Full sweep (backend={args.backend}): {len(triples)} combos")
        else:
            print(f"Full sweep: {len(triples)} (model, system, backend) combos")

        print(f"  Models:   {len(set(t[0] for t in triples))}")
        print(f"  Systems:  {sorted(set(t[1] for t in triples))}")
        print(f"  Backends: {sorted(set(t[2] for t in triples))}")
        print()

        if not triples:
            sys.exit(f"Error: no supported combos for backend '{args.backend}'")

        suffix = f"_{args.backend}" if args.backend else ""
        output = args.output or f"full_sweep{suffix}.csv"
        all_rows = []

        for i, (model, system, backend) in enumerate(triples):
            ms = model.split("/")[-1]
            gpu = system.upper().replace("_", " ")
            tag = f"[{i+1}/{len(triples)}] {ms} / {gpu} / {backend}: "
            print(f"\n{tag}probing...")
            rows = sweep_one_triple(model, system, backend, label=tag)
            all_rows.extend(rows)
            print(f"  {tag}{len(rows)} results (total: {len(all_rows)})")

            # Checkpoint every 10 triples
            if (i + 1) % 10 == 0 and all_rows:
                save_csv(all_rows, output)
                elapsed = time.time() - t0
                print(f"  [checkpoint] {elapsed/60:.1f} min elapsed")

        save_csv(all_rows, output)

    elif args.model and args.system:
        # ── Single or filtered ──
        ms = args.model.split("/")[-1]

        if args.backend:
            # One specific triple
            backends = [args.backend]
        else:
            # All supported backends for this (model, system)
            triples = load_support_triples()
            backends = sorted(set(be for m, s, be in triples if m == args.model and s == args.system))
            if not backends:
                # Fallback: try common backends
                backends = ["trtllm", "vllm", "sglang"]
            print(f"Backends for {ms}/{args.system}: {backends}")

        output = args.output or f"sweep_{ms}_{args.system}.csv"
        all_rows = []

        for backend in backends:
            print(f"\n{ms} / {args.system} / {backend}: probing...")
            rows = sweep_one_triple(args.model, args.system, backend)
            all_rows.extend(rows)

        save_csv(all_rows, output)

    else:
        parser.print_help()
        print("\nError: provide --full, or --model + --system")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
