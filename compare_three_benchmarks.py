import json
import os
from collections import defaultdict

BASE_DIR_OLD = os.path.join("Benchmarking_custom", "testing")
BASE_DIR_NEW1 = os.path.join("Benchmarking_custom", "testing_new1")

DATASETS = [
    ("General Cyber (100)", "prompts_100.jsonl", "benchmark_results_100.jsonl", "benchmark_results_100_new.jsonl", "benchmark_results_100_new1.jsonl"),
    ("Adversarial & Exploits (100)", "prompts_adversarial_100.jsonl", "benchmark_results_adversarial_100.jsonl", "benchmark_results_adversarial_100_new.jsonl", "benchmark_results_adversarial_100_new1.jsonl"),
    ("Cloud & IaC Security (100)", "prompts_cloud_native_100.jsonl", "benchmark_results_cloud_native_100.jsonl", "benchmark_results_cloud_native_100_new.jsonl", "benchmark_results_cloud_native_100_new1.jsonl"),
    ("Low-Level & Rev Eng (100)", "prompts_lowlevel_100.jsonl", "benchmark_results_lowlevel_100.jsonl", "benchmark_results_lowlevel_100_new.jsonl", "benchmark_results_lowlevel_100_new1.jsonl"),
    ("Multi-Lang AppSec (100)", "prompts_multilang_100.jsonl", "benchmark_results_multilang_100.jsonl", "benchmark_results_multilang_100_new.jsonl", "benchmark_results_multilang_100_new1.jsonl"),
]

def parse_jsonl(filepath):
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records

def main():
    print("=" * 95)
    print(" THREE-WAY BENCHMARK COMPARISON (Baseline vs Previous New vs Multi-Exemplar New1)")
    print("=" * 95)
    print(f"{'Dataset':<28} | {'OLD (Baseline)':<14} | {'NEW (Previous)':<14} | {'NEW1 (Multi-Exemplar)':<20}")
    print("-" * 95)

    tot_old_corr, tot_new_corr, tot_new1_corr = 0, 0, 0
    tot_old_cnt, tot_new_cnt, tot_new1_cnt = 0, 0, 0

    for label, p_file, old_f, new_f, new1_f in DATASETS:
        old_recs = parse_jsonl(os.path.join(BASE_DIR_OLD, old_f))
        new_recs = parse_jsonl(os.path.join(BASE_DIR_OLD, new_f))
        new1_recs = parse_jsonl(os.path.join(BASE_DIR_NEW1, new1_f))

        old_corr = sum(1 for r in old_recs if r.get("expected_domain") == r.get("predicted_expert"))
        new_corr = sum(1 for r in new_recs if r.get("expected_domain") == r.get("predicted_expert"))
        new1_corr = sum(1 for r in new1_recs if r.get("expected_domain") == r.get("predicted_expert"))

        old_cnt = len(old_recs)
        new_cnt = len(new_recs)
        new1_cnt = len(new1_recs)

        old_acc = (old_corr / old_cnt * 100) if old_cnt > 0 else 0.0
        new_acc = (new_corr / new_cnt * 100) if new_cnt > 0 else 0.0
        new1_acc = (new1_corr / new1_cnt * 100) if new1_cnt > 0 else 0.0

        tot_old_corr += old_corr
        tot_new_corr += new_corr
        tot_new1_corr += new1_corr

        tot_old_cnt += old_cnt
        tot_new_cnt += new_cnt
        tot_new1_cnt += new1_cnt

        new1_str = f"{new1_acc:6.2f}% ({new1_corr}/{new1_cnt})" if new1_cnt > 0 else "Not Run Yet"
        print(f"{label:<28} | {old_acc:6.2f}% ({old_corr}/{old_cnt}) | {new_acc:6.2f}% ({new_corr}/{new_cnt}) | {new1_str}")

    print("-" * 95)
    tot_old_acc = (tot_old_corr / tot_old_cnt * 100) if tot_old_cnt > 0 else 0.0
    tot_new_acc = (tot_new_corr / tot_new_cnt * 100) if tot_new_cnt > 0 else 0.0
    tot_new1_acc = (tot_new1_corr / tot_new1_cnt * 100) if tot_new1_cnt > 0 else 0.0
    new1_tot_str = f"{tot_new1_acc:6.2f}% ({tot_new1_corr}/{tot_new1_cnt})" if tot_new1_cnt > 0 else "Not Run Yet"
    print(f"{'TOTAL (AGGREGATE ALL 5)':<28} | {tot_old_acc:6.2f}%          | {tot_new_acc:6.2f}%          | {new1_tot_str}")
    print("=" * 95)

if __name__ == "__main__":
    main()
