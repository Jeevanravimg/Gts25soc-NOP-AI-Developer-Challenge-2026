# NOP AI Developer Challenge 2026
# Candidate Solution

This repository contains a private submission package for the NOP AI Developer Challenge 2026. It includes the working source code, configuration files, generated event outputs, evidence snapshots, and a Streamlit review dashboard for the prerecorded CCTV-style analytics task.

## Project scope

The solution performs end-to-end processing for package movement and zone-based behavioral event detection across four public scenarios:

- `S01_BASIC_GOODS`
- `S02_OCCLUSION_REVERSAL`
- `S03_DENSE_CROSSING`
- `S04_DWELL_QUEUE`

The system produces:

- annotated video outputs
- event CSV and JSONL records
- scenario-level counts
- analytics summaries
- evidence image snapshots
- a dashboard for interactive review

## Operating system tested

- Windows 11
- PowerShell
- Local Python virtual environment (`.venv`)

## Runtime and language

- Python 3.8.6 (verified in the project virtual environment)
- Primary stack: Python, OpenCV, NumPy, SciPy, Pandas, Pillow, Streamlit, Plotly, jsonschema
- Test tooling present in the environment: pytest

## Exact setup commands

From the project root:

```powershell
cd candidate-solution
python -m pip install -r requirements.txt
```

If using the repository virtual environment directly:

```powershell
cd "C:\Users\admin\Desktop\Gts25soc-NOP-AI-Developer-Challenge-2026"
.\.venv\Scripts\python.exe -m pip install -r candidate-solution\requirements.txt
```

## Exact run commands

Run the pipeline on a prerecorded input video:

```powershell
cd candidate-solution
python src/main.py --input <path-to-video.mp4> --config config/s01_basic_goods.json --output-dir output/S01_BASIC_GOODS
```

Run the interactive review dashboard:

```powershell
cd candidate-solution
streamlit run app.py
```

## Expected input format

The pipeline expects a prerecorded video file in a standard OpenCV-compatible format such as:

- `.mp4`
- `.mkv`
- `.avi`

The project also expects a scenario configuration file from `config/`, such as:

- `config/default_config.json`
- `config/s01_basic_goods.json`
- `config/s02_occlusion_reversal.json`
- `config/s03_dense_crossing.json`
- `config/s04_dwell_queue.json`

## Output locations

Scenario outputs are stored under `output/`:

- `output/<SCENARIO>/annotated.mp4`
- `output/<SCENARIO>/preview.webp`
- `output/<SCENARIO>/events.csv`
- `output/<SCENARIO>/events.jsonl`
- `output/<SCENARIO>/counts.csv`
- `output/<SCENARIO>/all_events.csv`
- `output/<SCENARIO>/analytics_summary.json`
- `output/<SCENARIO>/evidence/`

The consolidated event export is:

- `output/consolidated_events.csv`

## Hardware used

The project was developed and validated on a local Windows workstation using a standard Python virtual environment. The repository does not record an exact CPU/GPU model or dedicated ML hardware specification.

## Known limitations

- The implementation is designed for offline processing of prerecorded videos, not live camera streams.
- Detection and tracking logic are tuned for package-like objects in the provided surveillance challenge scenes.
- Zone geometry and thresholds are configuration-driven but not generalized to arbitrary unseen scenes without adjustment.
- The project does not include a separate training pipeline, retraining workflow, or custom model weights.
- `dwell_analytics` remains empty where the generated analytics summary contains no dwell data; no synthetic dwell metrics are added.
- The dashboard consumes generated output files from `output/` and does not re-run the video pipeline.

## Submission notes

This package is intended for private submission to the TSCI interview team and is not intended for public publishing in the starter repository.