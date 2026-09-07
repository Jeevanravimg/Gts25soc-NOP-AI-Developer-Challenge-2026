# Created by: Jeevan M G
# Date: 05-09-2026
# Explanation: This baseline evaluator checks output quality across scenarios by comparing expected and
# predicted event totals while preparing a compact summary report.

import os
import json
import csv
import time
import shutil
import subprocess
from pathlib import Path
from collections import Counter
import cv2

def load_truth(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def ensure_clean_dir(dir_path):
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)

def norm_event(scenario_id, event_type):
    return scenario_id.strip(), (event_type or "").strip().upper()

def main():
    base_output_dir = Path("output/baseline")
    scenarios_dir = Path("data/scenarios")
    videos_dir = Path("data/generated")
    truth_path = Path("data/ground_truth/public_event_truth.csv")

    scenarios = ["S01_BASIC_GOODS", "S02_OCCLUSION_REVERSAL", "S03_DENSE_CROSSING", "S04_DWELL_QUEUE"]
    
    truth_data = load_truth(truth_path)
    
    # Store results for final report
    results = []

    for scenario_id in scenarios:
        # Find video path
        video_path = videos_dir / f"{scenario_id}.mp4"
        scenario_json_path = list(scenarios_dir.glob(f"*{scenario_id.lower().replace('s0', '0')}*.json"))
        if not scenario_json_path:
            scenario_json_path = list(scenarios_dir.glob(f"*{scenario_id[1:3]}*.json"))
            
        if not video_path.exists() or not scenario_json_path:
            print(f"Skipping {scenario_id}: required files missing.")
            continue
            
        scenario_json_path = scenario_json_path[0]
        with open(scenario_json_path, "r", encoding="utf-8") as f:
            scenario_data = json.load(f)
            
        # Parse zones and create robust config
        zones = {}
        for name, rect in scenario_data.get("zones", {}).items():
            x1, y1, x2, y2 = rect
            zones[name] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
        config = {
            "camera_id": scenario_id,
            "min_area": 900,
            "max_match_distance": 80,
            "max_missed_frames": 15,
            "zones": zones,
            "intermediate_zones": [n for n in zones if "QUEUE" in n.upper()]
        }
        
        scenario_out_dir = base_output_dir / scenario_id[:3]
        ensure_clean_dir(scenario_out_dir)
        config_path = scenario_out_dir / "config.json"
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            
        # Get video metadata
        cap = cv2.VideoCapture(str(video_path))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Run baseline
        start_time = time.time()
        cmd = [
            "python", "starter/main.py",
            "--input", str(video_path),
            "--output-dir", str(scenario_out_dir),
            "--config", str(config_path)
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc_time = time.time() - start_time
        proc_fps = total_frames / proc_time if proc_time > 0 else 0
        
        # Calculate event metrics via evaluate_events.py logic
        expected_events = [r for r in truth_data if r["scenario_id"] == scenario_id]
        expected_counts = Counter(r["event_type"].upper() for r in expected_events)

        pred_counts = Counter()
        counts_csv = scenario_out_dir / "counts.csv"
        if counts_csv.exists():
            with open(counts_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pred_counts[row["event_type"].upper()] += int(row["count"])

        keys = sorted(set(expected_counts) | set(pred_counts))
        
        total_expected = 0
        total_predicted = 0
        total_correct = 0
        total_missed = 0
        total_extra = 0
        
        for key in keys:
            t = expected_counts[key]
            p = pred_counts[key]
            m = min(t, p)
            total_expected += t
            total_predicted += p
            total_correct += m
            total_missed += max(0, t - p)
            total_extra += max(0, p - t)
            
        precision = total_correct / total_predicted if total_predicted > 0 else 0.0
        recall = total_correct / total_expected if total_expected > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        
        results.append({
            "Scenario": scenario_id,
            "Expected": total_expected,
            "Predicted": total_predicted,
            "Matched/Correct": total_correct,
            "Missed": total_missed,
            "Extra": total_extra,
            "Count-based Precision": f"{precision:.4f}",
            "Count-based Recall": f"{recall:.4f}",
            "Count-based F1": f"{f1:.4f}",
            "Total Frames": total_frames,
            "Video FPS": f"{fps_video:.2f}",
            "Processing Time (s)": f"{proc_time:.2f}",
            "Processing FPS": f"{proc_fps:.2f}"
        })

    # Print Final Evaluation Report
    print("=" * 110)
    print("BASELINE EVALUATION REPORT")
    print("=" * 110)
    for res in results:
        print(f"\nScenario: {res['Scenario']}")
        print("-" * 50)
        print("--- Count-Based Accuracy Metrics ---")
        print(f"  Expected events        : {res['Expected']}")
        print(f"  Predicted events       : {res['Predicted']}")
        print(f"  Matched/Correct Count  : {res['Matched/Correct']}")
        print(f"  Missed Count           : {res['Missed']}")
        print(f"  Extra Count            : {res['Extra']}")
        print(f"  Count-based Precision  : {res['Count-based Precision']}")
        print(f"  Count-based Recall     : {res['Count-based Recall']}")
        print(f"  Count-based F1         : {res['Count-based F1']}")
        print("--- Performance Metrics ---")
        print(f"  Total frames           : {res['Total Frames']}")
        print(f"  Video FPS              : {res['Video FPS']}")
        print(f"  Processing time        : {res['Processing Time (s)']}s")
        print(f"  Processing FPS         : {res['Processing FPS']}")
        print("-" * 50)
    print("\nEvaluation successfully completed across all scenarios.")

if __name__ == "__main__":
    main()
