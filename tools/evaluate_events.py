import argparse
import csv
from collections import Counter
from pathlib import Path


def load_truth(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_candidate(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def norm_event(row):
    scenario = (row.get("scenario_id") or row.get("scenario") or "").strip()
    event = (row.get("event_type") or row.get("direction") or "").strip().upper()
    return scenario, event


def main():
    parser = argparse.ArgumentParser(description="Simple public event-count evaluator for candidate CSV output.")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate CSV. Must contain scenario_id and event_type/direction.")
    parser.add_argument("--truth", type=Path, default=Path("data/ground_truth/public_event_truth.csv"))
    args = parser.parse_args()

    truth = Counter(norm_event(r) for r in load_truth(args.truth))
    pred = Counter(norm_event(r) for r in load_candidate(args.candidate))

    keys = sorted(set(truth) | set(pred))
    tp = fp = fn = 0
    print("scenario,event_type,truth,predicted,matched")
    for key in keys:
        t = truth[key]
        p = pred[key]
        m = min(t, p)
        tp += m
        fp += max(0, p - t)
        fn += max(0, t - p)
        print(f"{key[0]},{key[1]},{t},{p},{m}")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print("\nPUBLIC EVENT-COUNT SCORE")
    print(f"matched={tp} false_positive={fp} missed={fn}")
    print(f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    print("Note: final interview scoring also checks tracking quality, evidence integrity, unseen footage, performance and live modification.")


if __name__ == "__main__":
    main()
