# NOP AI Developer Challenge 2026
# Technical Report

## 1. Problem selected

The selected task is generic goods/package movement intelligence for prerecorded surveillance-style video. The objective is to detect package-like objects, keep object identities stable across frames, reason about zone transitions, generate directional events, and preserve evidence for review.

The project targets the four challenge scenarios in the repository:

- `S01_BASIC_GOODS`
- `S02_OCCLUSION_REVERSAL`
- `S03_DENSE_CROSSING`
- `S04_DWELL_QUEUE`

## 2. Architecture and approach

The workflow is structured as a modular vision pipeline:

```text
Input Video
    ↓
Adaptive Detection
    ↓
Multi-object Tracking
    ↓
Zone / Geometry Processing
    ↓
Event Engine
    ↓
Evidence Generation
    ↓
Analytics Summary
    ↓
CSV / JSONL / Annotated Video / Review UI
```

The actual implementation is organized by the modules below:

- `src/detector.py` — adaptive detection and color-aware filtering
- `src/tracker.py` — Kalman-based tracker with object coasting and hue-gated matching
- `src/zones.py` — zone management and geometry handling
- `src/event_engine.py` — stateful A/B directional detection and queue/dwell logic
- `src/evidence.py` — event evidence snapshot capture
- `src/analytics.py` — queue and track analytics summarization
- `src/pipeline.py` — orchestrates detection, tracking, event generation, and outputs

This matches the project intent: process input video into structured outputs and evidence files, then use the review dashboard to inspect the final results.

## 3. Detector, model, and tracker used

### Detector

The project uses `AdaptiveDetector` from `src/detector.py`.

It combines:

- background subtraction via OpenCV MOG2
- morphology operations for noise suppression
- adaptive color saliency checks for synthetic challenge scenes
- dominant-hue extraction and color descriptor generation
- non-maximum suppression to suppress duplicate boxes

The detector is designed for package-like objects in structured surveillance imagery rather than a generic pretrained object detector.

### Tracker

The project uses `RobustKalmanTracker` from `src/tracker.py`.

The tracker uses a constant-velocity Kalman filter with:

- centroid and bounding-box state estimation
- IoU-aware cost matching
- hue-gated matching to reduce false associations across similar motion patterns
- coasting behavior for temporary occlusion or missed detections

This is an explicit choice for the challenge scenarios that involve reversal, dense crossing, and stop/resume behavior.

### Event logic

The event engine in `src/event_engine.py` uses a stateful journey model to detect:

- `A_TO_B`
- `B_TO_A`
- dwell threshold events
- queue occupancy metrics

The design explicitly tracks terminal-zone transitions and ignores duplicate re-triggering of the same transition path.

## 4. Why these choices were made

The project constraints are a good fit for a lightweight rule-based pipeline:

- The challenge data is prerecorded and camera geometry is fixed.
- Objects are package-like and occupy structured zones.
- The synthetic scenarios emphasize direction changes, queue behavior, and dense crossings rather than arbitrary generic object classes.
- The detector/tracker combination is effective for identity maintenance under partial occlusion and dense motion without requiring a large external model or weight package.
- The project keeps the same config-driven zone geometry approach used across scenarios and scales the geometry relative to runtime frame dimensions.

The implementation is therefore intentionally practical and deterministic for the provided benchmark conditions.

## 5. Evaluation method and measured results

The repository includes the public event-count evaluator at `tools/evaluate_events.py`.

This evaluator compares:

- `output/consolidated_events.csv`
- against `data/ground_truth/public_event_truth.csv`

It counts event tuples by `(scenario_id, event_type)` and reports precision, recall, and F1.

Measured project result from the repository evaluator:

```text
scenario,event_type,truth,predicted,matched
S01_BASIC_GOODS,A_TO_B,2,2,2
S01_BASIC_GOODS,B_TO_A,1,1,1
S02_OCCLUSION_REVERSAL,A_TO_B,2,2,2
S02_OCCLUSION_REVERSAL,B_TO_A,1,1,1
S03_DENSE_CROSSING,A_TO_B,3,3,3
S03_DENSE_CROSSING,B_TO_A,2,2,2
S04_DWELL_QUEUE,A_TO_B,3,3,3

PUBLIC EVENT-COUNT SCORE
matched=14 false_positive=0 missed=0
precision=1.0000 recall=1.0000 f1=1.0000
```

This is the actual measured overall result produced by the project’s evaluator and should be treated as the recorded public result for the challenge outputs in this repository.

## 6. Runtime performance

The generated analytics summaries contain source-frame rate values of `25.0` FPS for each scenario, for example:

- `S01_BASIC_GOODS`: `fps = 25.0`
- `S02_OCCLUSION_REVERSAL`: `fps = 25.0`
- `S03_DENSE_CROSSING`: `fps = 25.0`
- `S04_DWELL_QUEUE`: `fps = 25.0`

This is the source video FPS, not a processing-FPS benchmark. The project does not include a measured end-to-end processing FPS or wall-clock throughput table in the repository, so no separate processing benchmark is being claimed.

The resulting workflow is therefore best described as an offline video analytics pipeline for prerecorded footage.

## 7. Major failure cases

The actual project logic and outputs identify the most relevant limitations:

- dense crossings can still create ambiguous object matches when several package-like objects overlap closely;
- severe occlusion or identity reversals may require a long coasting period before the tracker re-stabilizes;
- the queue/dwell logic is scenario-specific and depends on the configuration values in `config/` and the generated scene geometry;
- the system is not general-purpose across unrelated object classes or truly unconstrained camera viewpoints;
- `dwell_analytics` is empty in the generated summaries when no dwell metrics are computed; the project does not fabricate missing dwell values.

## 8. What I would improve with more time

Given more time, the most useful improvements would be:

- more robust generalized detection for object types beyond the current package-focused benchmark;
- stronger occlusion handling and identity recovery for crowded scenes;
- more explicit event confidence and post-processing filtering to reduce edge ambiguity;
- additional calibration or dynamic zone adaptation for unseen similar camera setups;
- expanded analytics for queue dwell and occupancy stability across more scenarios.

These are improvement directions only; they are not claims of additional capabilities already implemented in the project.

## 9. AI development tools used

The repository reflects a standard local Python development workflow, including:

- Python 3.8.6 virtual environment
- VS Code / local project workspace
- pip-based dependency installation
- OpenCV and scientific Python stack for image processing and analysis
- Streamlit for the review dashboard
- Plotly for charting
- pytest for environment verification

No separate machine-learning training pipeline, training logs, or custom GPU environment are included in the submission package.

## 10. Third-party components and licenses

The project depends on third-party components, including:

- OpenCV
- NumPy
- SciPy
- pandas
- Pillow
- Streamlit
- Plotly
- jsonschema
- pytest

These libraries are used as upstream dependencies in the project environment. Their license terms are defined by their respective packages and are not reproduced here as a bundled project artifact.

The repository does not include any external trained model weights, vendor SDKs, or proprietary model binaries in the submission package.

## 11. Summary

This submission is a working, scenario-based video analytics pipeline that generates structured output artifacts from prerecorded CCTV-style video. It is grounded in actual project files, uses the generated outputs and evaluator from the repository, and records a measured public event-count result of:

- matched = 14
- false_positive = 0
- missed = 0
- precision = 1.0000
- recall = 1.0000
- f1 = 1.0000

The work is intended for private review and is not a public release of the full challenge solution.