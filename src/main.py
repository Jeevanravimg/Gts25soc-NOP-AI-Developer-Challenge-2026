# Created by: Jeevan M G
# Date: 05-09-2026
"""Command-line entry point for the candidate solution.

This version is structured to work when launched either as a module from the project root
(`python -m src.main`) or as a script directly from the `src` directory.
"""

import argparse
import json
from pathlib import Path

try:
    from src.pipeline import VisionPipeline
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from pipeline import VisionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NOP Pro+ Vision Intelligence Challenge - Candidate Solution"
    )
    parser.add_argument("--input", required=True, help="Path to input video file (MP4, MKV, AVI)")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON configuration. If not specified, auto-matches scenario.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save evidence, CSVs, JSONL, and annotated video",
    )
    parser.add_argument(
        "--scenario-id",
        default=None,
        help="Scenario ID (e.g., S01_BASIC_GOODS). Auto-inferred if omitted.",
    )
    parser.add_argument("--display", action="store_true", help="Display live annotated video window")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip rendering output MP4 video for maximum speed",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_config(input_path: str, config_path: str = None) -> tuple:
    """Finds or builds the best matching configuration based on the input video name."""
    video_stem = Path(input_path).stem.upper()

    scenario_map = {
        "S01": ("S01_BASIC_GOODS", "config/s01_basic_goods.json"),
        "S02": ("S02_OCCLUSION_REVERSAL", "config/s02_occlusion_reversal.json"),
        "S03": ("S03_DENSE_CROSSING", "config/s03_dense_crossing.json"),
        "S04": ("S04_DWELL_QUEUE", "config/s04_dwell_queue.json"),
    }

    inferred_scenario = "GENERIC_VIDEO"
    default_cfg = "config/default_config.json"

    for prefix, (scenario_id, cfg_file) in scenario_map.items():
        if prefix in video_stem:
            inferred_scenario = scenario_id
            default_cfg = cfg_file
            break

    target_config = config_path or default_cfg
    project_root = _project_root()
    cfg_path = Path(target_config)

    if not cfg_path.is_file():
        cfg_path = project_root / target_config
    if not cfg_path.is_file():
        cfg_path = project_root / "config" / "default_config.json"

    with open(cfg_path, "r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    return config_data, inferred_scenario


def main() -> None:
    args = parse_args()
    config_data, inferred_scenario = resolve_config(args.input, args.config)
    scenario_id = args.scenario_id or config_data.get("scenario_id", inferred_scenario)
    config_data["scenario_id"] = scenario_id

    print("\n========================================================")
    print(" NOP Pro+ Vision Intelligence - Pipeline Starting")
    print(f" Scenario:   {scenario_id}")
    print(f" Input:      {args.input}")
    print(f" Output Dir: {args.output_dir}")
    print("========================================================\n")

    pipeline = VisionPipeline(
        config=config_data,
        output_dir=args.output_dir,
        scenario_id=scenario_id,
    )

    result = pipeline.process_video(
        video_path=args.input,
        display=args.display,
        save_video=not args.no_video,
    )

    print("\n---------------- Processing Complete ----------------")
    print(f"Processed Frames: {result['processed_frames']}")
    print(f"Total Events:     {result['events_count']}")
    print("Event Breakdown:")
    for event_type, count in result["counts"].items():
        print(f"  - {event_type}: {count}")
    print(f"Outputs saved to: {args.output_dir}")
    print("========================================================\n")


if __name__ == "__main__":
    main()

# Explanation: This script resolves the config, launches the processing pipeline, and prints a summary
# of the generated output for the supplied video input.
