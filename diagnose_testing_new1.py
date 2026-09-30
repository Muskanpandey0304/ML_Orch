import json
import os
from collections import Counter

BASE_DIR = os.path.join("Benchmarking_custom", "testing_new1")

FILES = [
    ("General Cyber", "benchmark_results_100_new1.jsonl"),
    ("Adversarial", "benchmark_results_adversarial_100_new1.jsonl"),
    ("Cloud Native", "benchmark_results_cloud_native_100_new1.jsonl"),
    ("Low Level", "benchmark_results_lowlevel_100_new1.jsonl"),
    ("Multi-Lang", "benchmark_results_multilang_100_new1.jsonl"),
]

def analyze():
    print("=== DIAGNOSTIC REPORT FOR testing_new1 ===")
    total = 0
    correct = 0
    reranked_cnt = 0
    reranked_correct = 0
    misroutes = []
    
    for label, fname in FILES:
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                total += 1
                exp = d.get("expected_domain")
                pred = d.get("predicted_expert")
                is_corr = (exp == pred)
                if is_corr: correct += 1
                
                is_rr = d.get("reranked", False)
                if is_rr:
                    reranked_cnt += 1
                    if is_corr: reranked_correct += 1
                
                if not is_corr:
                    misroutes.append((label, d.get("id"), exp, pred, d.get("expert_conf")))

    print(f"Total Queries Evaluated: {total}")
    print(f"Total Correct: {correct} ({correct/total*100:.2f}%)")
    print(f"Queries Re-ranked by LLM: {reranked_cnt}")
    print(f"Re-ranked Queries Correct: {reranked_correct}")
    print(f"\nSample Misroutes (first 10):")
    for m in misroutes[:10]:
        print(f"  * [{m[0]}] {m[1]}: Expected={m[2]} Got={m[3]} Conf={m[4]}")

if __name__ == "__main__":
    analyze()
