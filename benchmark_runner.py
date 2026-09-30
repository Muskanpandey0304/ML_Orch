#!/usr/bin/env python3
"""
================================================================================
BENCHMARK RUNNER - Run prompt files ONE BY ONE against CyberMoE Orchestrator
================================================================================
Usage:
    python benchmark_runner.py                              # Interactive menu
    python benchmark_runner.py --dataset all               # Run all datasets
    python benchmark_runner.py --dataset all \\
        --orchestrator cyber_moe_orchestrator_optimised \\
        --output-dir Benchmarking_custom/baseline_run      # Baseline orchestrator
================================================================================
"""

import asyncio
import json
import os
import sys
import time
import argparse
import importlib
from typing import Any, Dict, List, Tuple

PROMPT_DIR = os.path.join("Benchmarking_custom", "testing")
OUTPUT_DIR = os.path.join("Benchmarking_custom", "testing_new1")  # default, overrideable

# Orchestrator module — overrideable via --orchestrator flag
ORCHESTRATOR_MODULE = "cyber_moe_orch_optimised"  # default

# Each tuple: (prompt_file, output_file, short_label)
DATASETS = [
    ("prompts_100.jsonl",              "benchmark_results_100_new1.jsonl",              "General Cyber (Baseline)"),
    ("prompts_adversarial_100.jsonl",  "benchmark_results_adversarial_100_new1.jsonl",  "Adversarial & Exploits"),
    ("prompts_cloud_native_100.jsonl", "benchmark_results_cloud_native_100_new1.jsonl", "Cloud & IaC Security"),
    ("prompts_lowlevel_100.jsonl",     "benchmark_results_lowlevel_100_new1.jsonl",     "Low-Level & Reverse Eng"),
    ("prompts_multilang_100.jsonl",    "benchmark_results_multilang_100_new1.jsonl",    "Multi-Language AppSec"),
]


def print_banner():
    print("\n" + "=" * 70)
    print("  CYBER MoE BENCHMARK RUNNER - One-by-One Mode")
    print("=" * 70)


def print_menu():
    print("\nAvailable Benchmark Datasets:")
    print("-" * 50)
    for i, (pf, _, label) in enumerate(DATASETS, 1):
        exists = "✅" if os.path.exists(os.path.join(PROMPT_DIR, pf)) else "❌"
        print(f"  [{i}] {exists} {label:30s} ({pf})")
    print(f"  [A] Run ALL datasets sequentially")
    print(f"  [Q] Quit")
    print("-" * 50)


async def run_single_benchmark(app: Any, prompt_file: str, output_file: str, label: str) -> Dict:
    """Run a single benchmark dataset and return summary stats."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    in_path = os.path.join(PROMPT_DIR, prompt_file)
    out_path = os.path.join(OUTPUT_DIR, output_file)

    if not os.path.exists(in_path):
        print(f"  ❌ File not found: {in_path}")
        return {}

    # Load all prompts
    prompts = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    total = len(prompts)
    print(f"\n{'=' * 70}")
    print(f"  🚀 RUNNING: {label}")
    print(f"  📄 Input : {in_path} ({total} prompts)")
    print(f"  📝 Output: {out_path}")
    print(f"{'=' * 70}\n")

    correct_routing = 0
    total_elapsed = 0.0
    total_tokens = 0
    errors = 0
    misroutes: List[Dict] = []  # Track misrouted prompts for debugging
    reranked_count = 0
    reranked_correct = 0
    reranked_overrides = 0
    reranked_override_correct = 0

    with open(out_path, "w", encoding="utf-8") as f_out:
        for idx, item in enumerate(prompts, 1):
            q_id = item.get("id", f"Q{idx}")
            expected_domain = item.get("domain", "?")
            prompt = item.get("prompt", "")

            try:
                start = time.perf_counter()
                res = await app.process_query(prompt)
                elapsed = time.perf_counter() - start

                predicted_expert = res.get("expert_id")
                is_correct = (predicted_expert == expected_domain)

                if is_correct:
                    correct_routing += 1
                    status = "✅"
                else:
                    status = "❌"
                    misroutes.append({
                        "id": q_id,
                        "expected": expected_domain,
                        "predicted": predicted_expert,
                        "conf": res.get("expert_conf"),
                        "prompt_preview": prompt[:80],
                    })

                if res.get("reranked", False):
                    reranked_overrides += 1
                    if is_correct:
                        reranked_override_correct += 1

                rerank_reason = res.get("rerank_reason", "unknown")
                if rerank_reason in ("llm_confirmed", "llm_override") or res.get("reranked", False) or rerank_reason.startswith("llm_override"):
                    reranked_count += 1
                    if is_correct:
                        reranked_correct += 1

                total_elapsed += res["final_res"]["elapsed"]
                total_tokens += res["final_res"]["tokens"]

                # Save result record
                record = {
                    "id": q_id,
                    "expected_domain": expected_domain,
                    "predicted_expert": predicted_expert,
                    "expert_conf": res.get("expert_conf"),
                    "route_tier": res.get("route"),
                    "variant_id": res.get("variant_id"),
                    "variant_conf": res.get("variant_conf"),
                    "reranked": res.get("reranked", False),
                    "rerank_reason": res.get("rerank_reason", "unknown"),
                    "plan_elapsed": res["plan_res"]["elapsed"],
                    "final_elapsed": res["final_res"]["elapsed"],
                    "final_tokens": res["final_res"]["tokens"],
                    "final_tps": res["final_res"]["tps"],
                    "plan_output": res["plan_res"]["answer"],
                    "final_output": res["final_res"]["answer"],
                }
                f_out.write(json.dumps(record) + "\n")

                # Progress line
                tier_short = res.get("route", "?")[:6]
                conf_str = f"{res.get('expert_conf', 0):.3f}" if res.get("expert_conf") else "N/A"
                print(f"  [{idx:3d}/{total}] {status} {q_id:12s} | Expected: {expected_domain} | Predicted: {predicted_expert or 'None':4s} | Conf: {conf_str} | Tier: {tier_short}")

            except Exception as e:
                errors += 1
                print(f"  [{idx:3d}/{total}] ⚠️  {q_id:12s} | ERROR: {str(e)[:60]}")
                # Write error record
                record = {
                    "id": q_id,
                    "expected_domain": expected_domain,
                    "predicted_expert": None,
                    "error": str(e),
                }
                f_out.write(json.dumps(record) + "\n")

    # ── Summary ──
    accuracy = (correct_routing / total) * 100 if total > 0 else 0
    avg_tps = (total_tokens / total_elapsed) if total_elapsed > 0 else 0

    summary = {
        "dataset": label,
        "total_prompts": total,
        "correct_routing": correct_routing,
        "accuracy_pct": round(accuracy, 2),
        "errors": errors,
        "total_tokens": total_tokens,
        "avg_tps": round(avg_tps, 1),
        "reranked_count": reranked_count,
        "reranked_correct": reranked_correct,
        "reranked_overrides": reranked_overrides,
        "reranked_override_correct": reranked_override_correct,
        "misroutes": misroutes,
    }

    print(f"\n{'─' * 70}")
    print(f"  📊 RESULTS: {label}")
    print(f"{'─' * 70}")
    print(f"  Routing Accuracy : {accuracy:.2f}%  ({correct_routing}/{total})")
    print(f"  Errors           : {errors}")
    print(f"  Avg Tokens/sec   : {avg_tps:.1f} tps")
    print(f"  LLM Verified     : {reranked_count} queries")
    print(f"  LLM Overrides    : {reranked_overrides} (correct: {reranked_override_correct})")
    print(f"  Output saved to  : {out_path}")

    if misroutes:
        print(f"\n  ⚠️  Misrouted Prompts ({len(misroutes)}):")
        for m in misroutes[:10]:  # Show first 10
            print(f"     • {m['id']:12s} Expected={m['expected']} Got={m['predicted']} (conf={m['conf']:.3f})")
            print(f"       \"{m['prompt_preview']}...\"")
        if len(misroutes) > 10:
            print(f"     ... and {len(misroutes) - 10} more (check output file for details)")

    print(f"{'─' * 70}\n")
    return summary


async def main():
    global OUTPUT_DIR  # allow override from CLI args

    parser = argparse.ArgumentParser(description="CyberMoE Benchmark Runner")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset number (1-5) or 'all'. If omitted, shows interactive menu."
    )
    parser.add_argument(
        "--orchestrator", type=str, default=ORCHESTRATOR_MODULE,
        help="Python module name of the orchestrator to benchmark "
             "(e.g. cyber_moe_orchestrator_optimised). Default: cyber_moe_orch_optimised"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Folder to save benchmark result files. "
             "Default: Benchmarking_custom/testing_new1"
    )
    parser.add_argument(
        "--server-url", type=str, default=None,
        help="Target server URL (e.g. http://127.0.0.1:8000/v1 or http://10.10.1.81:8001/v1)"
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Model name (e.g. bg-ft-experimental or bg-ft-774)"
    )
    args = parser.parse_args()

    if args.server_url:
        os.environ["SERVER_URL"] = args.server_url
    if args.model_name:
        os.environ["MODEL_NAME"] = args.model_name

    # ── Apply output-dir override ──
    if args.output_dir:
        OUTPUT_DIR = args.output_dir

    # ── Dynamically import the selected orchestrator ──
    orch_module_name = args.orchestrator
    if orch_module_name.endswith(".py"):
        orch_module_name = orch_module_name[:-3]

    print(f"\n  📦 Loading orchestrator module: {orch_module_name}")
    try:
        orch_module = importlib.import_module(orch_module_name)
        CyberMoEOrchestrator = orch_module.CyberMoEOrchestrator
    except ModuleNotFoundError as e:
        if getattr(e, "name", None) == orch_module_name:
            print(f"  ❌ Cannot find module '{orch_module_name}.py' in current directory.")
        else:
            print(f"  ❌ Missing required dependency inside '{orch_module_name}.py': {e}")
        return
    except Exception as e:
        print(f"  ❌ Failed to import module '{orch_module_name}': {e}")
        return

    print_banner()
    print(f"  🔧 Orchestrator : {orch_module_name}")
    print(f"  📂 Output Dir   : {OUTPUT_DIR}")

    # Initialize orchestrator once (loads sentence-transformers model)
    print("\n  ⏳ Loading orchestrator (sentence-transformers + expert taxonomy)...")
    app = CyberMoEOrchestrator()
    print("  ✅ Orchestrator ready!\n")

    summaries: List[Dict] = []

    try:
        if args.dataset:
            # CLI mode
            if args.dataset.lower() == "all":
                indices = list(range(len(DATASETS)))
            else:
                idx = int(args.dataset) - 1
                if idx < 0 or idx >= len(DATASETS):
                    print(f"  ❌ Invalid dataset number. Choose 1-{len(DATASETS)}.")
                    return
                indices = [idx]

            for i in indices:
                pf, of, label = DATASETS[i]
                s = await run_single_benchmark(app, pf, of, label)
                if s:
                    summaries.append(s)
        else:
            # Interactive menu mode
            while True:
                print_menu()
                choice = input("\n  Select dataset > ").strip().upper()

                if choice == "Q":
                    print("  Goodbye!")
                    break
                elif choice == "A":
                    for pf, of, label in DATASETS:
                        s = await run_single_benchmark(app, pf, of, label)
                        if s:
                            summaries.append(s)
                    break
                elif choice.isdigit() and 1 <= int(choice) <= len(DATASETS):
                    idx = int(choice) - 1
                    pf, of, label = DATASETS[idx]
                    s = await run_single_benchmark(app, pf, of, label)
                    if s:
                        summaries.append(s)

                    cont = input("\n  Run another dataset? (y/n) > ").strip().lower()
                    if cont != "y":
                        break
                else:
                    print("  ❌ Invalid choice. Try again.")

    except KeyboardInterrupt:
        print("\n\n  ⚠️  Interrupted by user.")
    finally:
        await app.shutdown()

    # ── Final Combined Summary ──
    if len(summaries) > 1:
        print(f"\n{'=' * 70}")
        print(f"  📋 COMBINED SUMMARY ACROSS ALL RUNS")
        print(f"{'=' * 70}")
        print(f"  {'Dataset':<30s} {'Accuracy':>10s} {'Errors':>8s} {'Avg TPS':>10s}")
        print(f"  {'─' * 58}")
        for s in summaries:
            print(f"  {s['dataset']:<30s} {s['accuracy_pct']:>9.2f}% {s['errors']:>8d} {s['avg_tps']:>9.1f}")

        total_correct = sum(s["correct_routing"] for s in summaries)
        total_prompts = sum(s["total_prompts"] for s in summaries)
        overall_acc = (total_correct / total_prompts) * 100 if total_prompts > 0 else 0
        print(f"  {'─' * 58}")
        print(f"  {'OVERALL':<30s} {overall_acc:>9.2f}% ({total_correct}/{total_prompts})")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    asyncio.run(main())
