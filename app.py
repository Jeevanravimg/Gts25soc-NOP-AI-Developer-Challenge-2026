# Created by: Jeevan M G
# Date: 05-09-2026
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
TRUTH_PATH = ROOT / "data" / "ground_truth" / "public_event_truth.csv"


st.set_page_config(
    page_title="AI Video Analytics & Behavioral Event Detection",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@st.cache_data
def discover_scenarios() -> List[Path]:
    if not OUTPUT_DIR.exists():
        return []
    scenario_dirs = []
    for entry in sorted(OUTPUT_DIR.iterdir(), key=lambda p: p.name):
        if entry.is_dir() and entry.name not in {"__pycache__"}:
            scenario_dirs.append(entry)
    return scenario_dirs


@st.cache_data
def load_scenario_summary(scenario_dir: str) -> Dict[str, Any]:
    path = Path(scenario_dir) / "analytics_summary.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


@st.cache_data
def load_event_csv(scenario_dir: str) -> pd.DataFrame:
    scenario_path = Path(scenario_dir)
    candidate_names = ["all_events.csv", "events.csv"]
    for name in candidate_names:
        path = scenario_path / name
        if path.exists():
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    df = df.copy()
                    df["scenario_id"] = df.get("scenario_id", scenario_path.name)
                    df["scenario_name"] = scenario_path.name
                    return df
            except Exception:
                continue
    return pd.DataFrame()


@st.cache_data
def load_jsonl_events(scenario_dir: str) -> List[Dict[str, Any]]:
    path = Path(scenario_dir) / "events.jsonl"
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return records


@st.cache_data
def load_counts_csv(scenario_dir: str) -> pd.DataFrame:
    path = Path(scenario_dir) / "counts.csv"
    if not path.exists():
        return pd.DataFrame(columns=["event_type", "count"])
    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame(columns=["event_type", "count"])
        return df
    except Exception:
        return pd.DataFrame(columns=["event_type", "count"])


@st.cache_data
def load_global_events() -> pd.DataFrame:
    scenario_dirs = discover_scenarios()
    frames: List[pd.DataFrame] = []
    for scenario_dir in scenario_dirs:
        df = load_event_csv(str(scenario_dir))
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.copy()
    for col in ["event_type", "scenario_id"]:
        if col in combined.columns:
            combined[col] = combined[col].fillna("")
    combined["event_type"] = combined.get("event_type", "").astype(str).str.upper()
    if "track_id" in combined.columns:
        combined["track_id"] = pd.to_numeric(combined["track_id"], errors="coerce")
    if "confidence" in combined.columns:
        combined["confidence"] = pd.to_numeric(combined["confidence"], errors="coerce")
    return combined


@st.cache_data
def load_truth_rows() -> List[Dict[str, Any]]:
    if not TRUTH_PATH.exists():
        return []
    try:
        with open(TRUTH_PATH, "r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


@st.cache_data
def compute_public_event_metrics() -> Dict[str, Any]:
    truth_rows = load_truth_rows()
    candidate_df = load_global_events()

    truth_counter: Counter[tuple[str, str]] = Counter()
    pred_counter: Counter[tuple[str, str]] = Counter()
    public_event_types = {"A_TO_B", "B_TO_A"}

    if candidate_df.empty:
        metrics = {
            "matched": 0,
            "false_positive": 0,
            "missed": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "status": "No candidate event data available",
        }
        return metrics

    for row in truth_rows:
        scenario = str(row.get("scenario_id") or row.get("scenario") or "").strip()
        event_type = str(row.get("event_type") or row.get("direction") or "").strip().upper()
        if scenario and event_type in public_event_types:
            truth_counter[(scenario, event_type)] += 1

    for _, row in candidate_df.iterrows():
        scenario = str(row.get("scenario_id") or row.get("scenario") or "").strip()
        event_type = str(row.get("event_type") or row.get("direction") or "").strip().upper()

        if scenario and event_type in public_event_types:
            pred_counter[(scenario, event_type)] += 1

    keys = sorted(set(truth_counter) | set(pred_counter))
    tp = fp = fn = 0
    for key in keys:
        truth_count = truth_counter.get(key, 0)
        pred_count = pred_counter.get(key, 0)
        matched = min(truth_count, pred_count)
        tp += matched
        fp += max(0, pred_count - truth_count)
        fn += max(0, truth_count - pred_count)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "matched": tp,
        "false_positive": fp,
        "missed": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "status": "Computed from public event truth and generated candidate output",
    }


@st.cache_data
def build_scenario_overview() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario_dir in discover_scenarios():
        summary = load_scenario_summary(str(scenario_dir))
        queue = summary.get("queue_analytics", {}) if isinstance(summary, dict) else {}
        speed_map = summary.get("track_speeds_px_per_sec", {}) if isinstance(summary, dict) else {}
        event_df = load_event_csv(str(scenario_dir))

        rows.append(
            {
                "Scenario": scenario_dir.name,
                "Source FPS": summary.get("fps", 0.0),
                "Frames/Samples": _safe_int(queue.get("samples", 0), 0),
                "Tracks": len(speed_map) if isinstance(speed_map, dict) else 0,
                "Events": len(event_df),
                "Peak Queue": _safe_int(queue.get("peak_occupancy", 0), 0),
                "Mean Queue": _safe_float(queue.get("mean_occupancy", 0.0), 0.0),
                "Scenario Directory": str(scenario_dir),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def get_case_path(scenario_name: str) -> Optional[Path]:
    if not scenario_name:
        return None
    for scenario_dir in discover_scenarios():
        if scenario_dir.name == scenario_name:
            return scenario_dir
    return None


@st.cache_data
def get_event_type_breakdown(scenario_name: str) -> pd.DataFrame:
    scenario_dir = get_case_path(scenario_name)
    if not scenario_dir:
        return pd.DataFrame(columns=["Event Type", "Count"])
    event_df = load_event_csv(str(scenario_dir))
    if event_df.empty:
        return pd.DataFrame(columns=["Event Type", "Count"])
    series = event_df.get("event_type", pd.Series(dtype=str)).astype(str).str.upper()
    counts = series.value_counts().reset_index()
    counts.columns = ["Event Type", "Count"]
    return counts.sort_values("Count", ascending=False).reset_index(drop=True)


@st.cache_data
def get_track_speed_frame(scenario_name: str) -> pd.DataFrame:
    scenario_dir = get_case_path(scenario_name)
    if not scenario_dir:
        return pd.DataFrame(columns=["Track ID", "Speed (px/sec)"])
    summary = load_scenario_summary(str(scenario_dir))
    speed_map = summary.get("track_speeds_px_per_sec", {}) if isinstance(summary, dict) else {}
    if not isinstance(speed_map, dict):
        return pd.DataFrame(columns=["Track ID", "Speed (px/sec)"])
    rows = [{"Track ID": int(track_id), "Speed (px/sec)": _safe_float(speed, 0.0)} for track_id, speed in speed_map.items()]
    if not rows:
        return pd.DataFrame(columns=["Track ID", "Speed (px/sec)"])
    return pd.DataFrame(rows).sort_values("Track ID").reset_index(drop=True)


def render_metric_card(label: str, value: Any, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div style="background: rgba(14, 17, 23, 0.72); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 0.8rem; min-height: 110px;">
            <div style="font-size: 0.76rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em;">{label}</div>
            <div style="font-size: 1.9rem; font-weight: 700; margin-top: 0.5rem; color: #f8fafc;">{value}</div>
            <div style="font-size: 0.72rem; color: #cbd5e1; margin-top: 0.3rem;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart_or_table(title: str, df: pd.DataFrame, x_col: str, y_col: str) -> None:
    if df.empty:
        st.info(f"No data available for {title}.")
        return

    if px is not None:
        fig = px.bar(df, x=x_col, y=y_col, title=title, color_discrete_sequence=["#4f46e5"])
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0f172a"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df.set_index(x_col)[y_col])


def render_sidebar() -> tuple[str, List[str]]:
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Choose a section",
        [
            "Dashboard",
            "Scenario Analysis",
            "Event Explorer",
            "Evidence Review",
            "System Architecture",
            "About / Technical Summary",
        ],
        index=0,
    )

    scenario_dirs = discover_scenarios()
    scenario_names = [scenario_dir.name for scenario_dir in scenario_dirs]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Scenario selector")
    if scenario_names:
        selected_scenario = st.sidebar.selectbox("Scenario", scenario_names, index=0)
    else:
        selected_scenario = ""
        st.sidebar.warning("No scenario folders were discovered under the output directory.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Output directory status")
    if OUTPUT_DIR.exists():
        st.sidebar.success(f"Output directory ready: {OUTPUT_DIR.name}")
    else:
        st.sidebar.error(f"Missing output directory: {OUTPUT_DIR.name}")
    st.sidebar.caption(f"Scenarios discovered: {len(scenario_names)}")

    if st.sidebar.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

    return page, scenario_names, selected_scenario


def render_dashboard() -> None:
    st.markdown("# AI Video Analytics & Behavioral Event Detection")
    st.caption("Computer Vision • Object Tracking • Zone Analytics • Event Intelligence")
    st.markdown("<div style='display:flex; align-items:center; gap:10px; margin-bottom:1rem;'><span style='display:inline-block; width:10px; height:10px; border-radius:50%; background:#22c55e; box-shadow:0 0 12px #22c55e;'></span><span style='font-size:0.8rem; color:#22c55e; font-weight:600;'>Pipeline Ready</span></div>", unsafe_allow_html=True)

    scenario_df = build_scenario_overview()
    global_events_df = load_global_events()
    public_metrics = compute_public_event_metrics()

    if scenario_df.empty:
        st.warning("No scenario output folders were found in the project output directory. Run the pipeline first to generate analytics outputs.")
        return

    total_detected_events = int(len(global_events_df))
    total_a_to_b = int(global_events_df.get("event_type", pd.Series(dtype=str)).astype(str).str.upper().eq("A_TO_B").sum()) if not global_events_df.empty else 0
    total_b_to_a = int(global_events_df.get("event_type", pd.Series(dtype=str)).astype(str).str.upper().eq("B_TO_A").sum()) if not global_events_df.empty else 0
    peak_queue = int(scenario_df["Peak Queue"].max()) if not scenario_df.empty else 0
    total_tracks = int(scenario_df["Tracks"].sum()) if not scenario_df.empty else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        render_metric_card("Scenarios Processed", len(scenario_df), "Discovered output folders")
    with col2:
        render_metric_card("Total Detected Events", total_detected_events, "Events read from output CSV exports")
    with col3:
        render_metric_card("A → B Events", total_a_to_b, "Directional transition events")
    with col4:
        render_metric_card("B → A Events", total_b_to_a, "Reverse-direction transitions")
    with col5:
        render_metric_card("Peak Queue Occupancy", peak_queue, "Highest queue occupancy across scenarios")
    with col6:
        render_metric_card("Total Tracks Observed", total_tracks, "Tracked object counts from analytics summaries")

    if truth_rows := load_truth_rows():
        st.info("Public evaluation metrics are computed dynamically from the repository truth file and the generated output events.")
        st.metric("Public Precision", f"{public_metrics['precision']:.4f}", help="Precision from public truth counts")
        st.metric("Public Recall", f"{public_metrics['recall']:.4f}", help="Recall from public truth counts")
        st.metric("Public F1", f"{public_metrics['f1']:.4f}", help="F1 score from public truth counts")
        st.metric("Matched / False Positive / Missed", f"{public_metrics['matched']} / {public_metrics['false_positive']} / {public_metrics['missed']}", help="TP / FP / FN counts")
    else:
        st.info("No public truth file was found; the dashboard is showing detected output counts from the generated scenario exports.")

    st.markdown("### Scenario Overview")
    scenario_table = scenario_df.copy()
    scenario_table = scenario_table[["Scenario", "Source FPS", "Frames/Samples", "Tracks", "Events", "Peak Queue", "Mean Queue"]]
    st.dataframe(scenario_table, use_container_width=True, hide_index=True)

    st.markdown("### Events by Scenario")
    event_counts = global_events_df["scenario_id"].fillna("").astype(str).value_counts().reset_index()
    event_counts.columns = ["Scenario", "Events"]
    chart_df = event_counts.sort_values("Scenario")
    if px is not None:
        fig = px.bar(chart_df, x="Scenario", y="Events", title="Events by Scenario", color_discrete_sequence=["#2563eb"])
        fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(chart_df.set_index("Scenario")["Events"])

    st.markdown("### Event Type Distribution")
    event_type_counts = global_events_df.get("event_type", pd.Series(dtype=str)).astype(str).str.upper().value_counts().reset_index()
    event_type_counts.columns = ["Event Type", "Count"]
    if px is not None:
        fig = px.pie(event_type_counts, names="Event Type", values="Count", title="Event Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(event_type_counts.set_index("Event Type")["Count"])

    st.markdown("### Track Speed Distribution")
    speed_rows: List[Dict[str, Any]] = []
    for scenario_dir in discover_scenarios():
        summary = load_scenario_summary(str(scenario_dir))
        speed_map = summary.get("track_speeds_px_per_sec", {}) if isinstance(summary, dict) else {}
        if isinstance(speed_map, dict):
            for track_id, speed in speed_map.items():
                speed_rows.append({"Scenario": scenario_dir.name, "Track ID": str(track_id), "Speed (px/sec)": _safe_float(speed, 0.0)})
    if speed_rows:
        speed_df = pd.DataFrame(speed_rows)
        if px is not None:
            fig = px.histogram(speed_df, x="Speed (px/sec)", nbins=20, title="Track Speed Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(speed_df["Speed (px/sec)"])
    else:
        st.info("No track-speed data is available for the discovered scenarios.")


def render_scenario_analysis(selected_scenario: str) -> None:
    st.title("Scenario Analysis")
    if not selected_scenario:
        st.warning("Select a scenario from the sidebar to inspect generated outputs.")
        return

    scenario_dir = get_case_path(selected_scenario)
    if not scenario_dir:
        st.warning("The selected scenario directory was not found.")
        return

    summary = load_scenario_summary(str(scenario_dir))
    event_df = load_event_csv(str(scenario_dir))
    speed_df = get_track_speed_frame(selected_scenario)
    breakdown_df = get_event_type_breakdown(selected_scenario)
    queue = summary.get("queue_analytics", {}) if isinstance(summary, dict) else {}
    track_speed_map = summary.get("track_speeds_px_per_sec", {}) if isinstance(summary, dict) else {}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Scenario", selected_scenario)
    with col2:
        st.metric("Source FPS", f"{summary.get('fps', 0.0):.1f}")
    with col3:
        st.metric("Frames/Samples", _safe_int(queue.get('samples', 0), 0))
    with col4:
        st.metric("Tracked Objects", len(track_speed_map) if isinstance(track_speed_map, dict) else 0)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Total Events", len(event_df))
    with col6:
        st.metric("Peak Queue", _safe_int(queue.get('peak_occupancy', 0), 0))
    with col7:
        st.metric("Mean Queue", _safe_float(queue.get('mean_occupancy', 0.0), 0.0))
    with col8:
        st.metric("Scenario Output", scenario_dir.name)

    st.markdown("### Track Speed Information")
    if speed_df.empty:
        st.info("No speed data is available in analytics_summary.json for this scenario.")
    else:
        st.dataframe(speed_df, use_container_width=True, hide_index=True)

    st.markdown("### Event Breakdown")
    if breakdown_df.empty:
        st.info("No events are available in the scenario output CSV for this scenario.")
    else:
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    preview_path = scenario_dir / "preview.webp"
    if preview_path.exists():
        st.markdown("### Scenario Preview")
        st.image(str(preview_path), use_container_width=True)
    else:
        st.caption("No preview.webp present for this scenario.")

    annotated_video = scenario_dir / "annotated.mp4"
    if annotated_video.exists():
        st.markdown("### Annotated Video")
        try:
            st.video(str(annotated_video))
        except Exception:
            st.warning("The browser preview could not render this MP4 directly. The generated preview.webp and evidence images are available below for review.")
            if preview_path.exists():
                st.image(str(preview_path), use_container_width=True)


def render_event_explorer() -> None:
    st.title("Event Explorer")
    global_events_df = load_global_events()
    if global_events_df.empty:
        st.warning("No event data was found in the output folders.")
        return

    df = global_events_df.copy()
    for col in ["scenario_id", "event_type", "object_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].astype(str).str.upper()
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    if "track_id" in df.columns:
        df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")

    st.sidebar.markdown("### Filters")
    scenario_options = ["All"] + sorted(df["scenario_id"].dropna().astype(str).unique().tolist())
    scenario_filter = st.sidebar.selectbox("Scenario", scenario_options, index=0)
    if scenario_filter != "All":
        df = df[df["scenario_id"].astype(str) == scenario_filter]

    event_type_options = ["All"] + sorted(df.get("event_type", pd.Series(dtype=str)).astype(str).str.upper().dropna().unique().tolist())
    event_type_filter = st.sidebar.selectbox("Event type", event_type_options, index=0)
    if event_type_filter != "All":
        df = df[df.get("event_type", pd.Series(dtype=str)).astype(str).str.upper() == event_type_filter]

    if "track_id" in df.columns:
        valid_tracks = sorted(pd.to_numeric(df["track_id"], errors="coerce").dropna().astype(int).unique().tolist())
        track_options = ["All"] + [str(value) for value in valid_tracks]
        track_filter = st.sidebar.selectbox("Track ID", track_options, index=0)
        if track_filter != "All":
            df = df[pd.to_numeric(df["track_id"], errors="coerce").astype(str) == str(track_filter)]

    if "confidence" in df.columns:
        min_conf = float(df["confidence"].dropna().min()) if not df["confidence"].dropna().empty else 0.0
        max_conf = float(df["confidence"].dropna().max()) if not df["confidence"].dropna().empty else 1.0
        conf_min = st.sidebar.slider("Minimum confidence", min_value=min_conf, max_value=max_conf, value=min_conf, step=0.01)
        df = df[df["confidence"].fillna(-1) >= conf_min]

    if df.empty:
        st.warning("No records match the selected filters.")
        return

    available_fields = [
        col for col in ["scenario_id", "event_type", "track_id", "observed_at_seconds", "confidence", "count", "object_type"]
        if col in df.columns
    ]
    if not available_fields:
        available_fields = list(df.columns)

    st.dataframe(df[available_fields], use_container_width=True, hide_index=True)

    st.markdown("### Event Details")
    event_options = sorted(df.get("event_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    if event_options:
        event_id = st.selectbox("Select event", event_options)
        selected = df[df.get("event_id", pd.Series(dtype=str)).astype(str) == event_id].iloc[0]

        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.write(f"**Scenario:** {selected.get('scenario_id', 'N/A')}")
            st.write(f"**Event type:** {selected.get('event_type', 'N/A')}")
            st.write(f"**Track ID:** {selected.get('track_id', 'N/A')}")
        with detail_col2:
            st.write(f"**Timestamp (s):** {selected.get('observed_at_seconds', 'N/A')}")
            st.write(f"**Confidence:** {selected.get('confidence', 'N/A')}")
            st.write(f"**Object type:** {selected.get('object_type', 'N/A')}")

        evidence_path_value = selected.get("evidence_image") or selected.get("image_path") or selected.get("evidence")
        if isinstance(evidence_path_value, dict):
            evidence_path_value = evidence_path_value.get("image_path") or evidence_path_value.get("path")

        if isinstance(evidence_path_value, str) and evidence_path_value:
            scenario_dir = get_case_path(str(selected.get("scenario_id", "")))
            if scenario_dir:
                image_path = scenario_dir / evidence_path_value.replace("/", "\\") if "\\" in evidence_path_value else scenario_dir / evidence_path_value
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.warning(f"The matching evidence file was not found: {image_path.name}")
            else:
                st.warning("The scenario directory could not be mapped for this evidence image.")
    else:
        st.info("No event IDs are available for selection in the current filtered dataset.")


def render_evidence_review(selected_scenario: str) -> None:
    st.title("Evidence Review")
    st.markdown("### Visual Evidence Generated by the Pipeline")

    if not selected_scenario:
        st.warning("Choose a scenario to review generated evidence snapshots.")
        return

    scenario_dir = get_case_path(selected_scenario)
    if not scenario_dir:
        st.warning("The selected scenario could not be found.")
        return

    evidence_dir = scenario_dir / "evidence"
    if not evidence_dir.exists():
        st.warning(f"No evidence directory exists for {selected_scenario}.")
        return

    evidence_files = sorted(
        [path for path in evidence_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}],
        key=lambda p: p.name,
    )

    if not evidence_files:
        st.warning(f"No visual evidence snapshots were generated for {selected_scenario}.")
        return

    event_jsonl = load_jsonl_events(str(scenario_dir))
    event_lookup: Dict[str, Dict[str, Any]] = {}
    for record in event_jsonl:
        event_id = record.get("event_id")
        if event_id:
            event_lookup[str(event_id)] = record

    cols = st.columns(3)
    for idx, image_path in enumerate(evidence_files):
        event_id = image_path.name.split("_")
        event_key = event_id[0] if event_id else ""
        event_record = event_lookup.get(event_key, {})
        event_type = event_record.get("event_type") or "Unknown"
        track_id = event_record.get("track_id")
        timestamp = event_record.get("attributes", {}).get("video_time_seconds") if isinstance(event_record.get("attributes"), dict) else None
        with cols[idx % 3]:
            st.image(str(image_path), use_container_width=True)
            st.caption(f"Event type: {event_type}")
            st.caption(f"Track ID: {track_id}")
            st.caption(f"Timestamp: {timestamp}")
            if event_record:
                st.code(json.dumps({
                    "scenario_id": event_record.get("scenario_id"),
                    "event_id": event_record.get("event_id"),
                    "event_type": event_record.get("event_type"),
                    "track_id": event_record.get("track_id"),
                    "confidence": event_record.get("confidence"),
                    "attributes": event_record.get("attributes"),
                }, indent=2, default=str), language="json")


def render_system_architecture() -> None:
    st.title("System Architecture")
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.8)); border: 1px solid rgba(148,163,184,0.25); border-radius: 18px; padding: 1.2rem; color: #e2e8f0;">
            <div style="text-align:center; font-weight:700; font-size:1.2rem; margin-bottom:0.7rem;">Project Flow</div>
            <div style="text-align:center;">Input Video</div>
            <div style="text-align:center;">↓</div>
            <div style="text-align:center;">Object Detection</div>
            <div style="text-align:center;">↓</div>
            <div style="text-align:center;">Object Tracking</div>
            <div style="text-align:center;">↓</div>
            <div style="text-align:center;">Zone / Geometry Processing</div>
            <div style="text-align:center;">↓</div>
            <div style="text-align:center;">Behavior / Event Engine</div>
            <div style="text-align:center;">↓</div>
            <div style="text-align:center;">Evidence Generation</div>
            <div style="text-align:center;">↓</div>
            <div style="text-align:center;">Analytics</div>
            <div style="text-align:center;">↓</div>
            <div style="text-align:center;">CSV / JSON / Evidence Images / Annotated Video</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Configurable components")
    st.markdown(
        """
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148,163,184,0.25); border-radius: 14px; padding: 1rem; margin-top: 0.8rem;">
            <div style="text-align:center;">Configuration</div>
            <div style="text-align:center;">↓</div>
            <div style="text-align:center;">Zones + Thresholds + Tracker Parameters</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Functional explanation")
    st.markdown(
        """
        - Detection identifies relevant objects in each frame.
        - Tracker maintains object identities across frames and handles motion continuity.
        - Zones define spatial regions that anchor directional logic and occupancy summaries.
        - Event engine identifies movement and behavior transitions between zones.
        - Evidence module records visual proof snapshots associated with events.
        - Analytics module summarizes track behavior, queue occupancy, and scenario metrics.
        - Outputs support review, scenario comparison, and downstream VMS integration.
        """
    )


def render_about_summary() -> None:
    st.title("About / Technical Summary")
    st.markdown("### Project")
    st.markdown("**AI Video Analytics & Behavioral Event Detection**")

    st.markdown("### Core capabilities")
    st.markdown(
        """
        - Object detection
        - Multi-object tracking
        - Configurable zone geometry
        - Directional transition events
        - Occlusion/reversal handling
        - Dense-crossing handling
        - Queue occupancy analytics
        - Evidence snapshot generation
        - JSON/CSV event export
        - Annotated video generation
        """
    )

    st.markdown("### Meaningful improvement")
    st.markdown(
        """
        Zone geometry is defined relative to a reference resolution, and runtime frame dimensions are used to scale the configured zone geometry. This improves robustness when the same configured zones are applied across different video resolutions.
        """
    )

    st.markdown("### Outputs and review workflow")
    st.markdown(
        """
        - Evidence outputs in each scenario directory
        - Structured event records in CSV and JSONL formats
        - Analytics summaries with FPS, queue metrics, and track-speed statistics
        - Scenario-based evaluation using the project's public event truth data when available
        """
    )


def main() -> None:
    page, scenario_names, selected_scenario = render_sidebar()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Scenario Analysis":
        render_scenario_analysis(selected_scenario)
    elif page == "Event Explorer":
        render_event_explorer()
    elif page == "Evidence Review":
        render_evidence_review(selected_scenario)
    elif page == "System Architecture":
        render_system_architecture()
    else:
        render_about_summary()

    if selected_scenario and selected_scenario in scenario_names:
        scenario_dir = get_case_path(selected_scenario)
        if scenario_dir:
            video_path = scenario_dir / "annotated.mp4"
            if video_path.exists():
                st.markdown("---")
                st.subheader(f"Annotated Video: {selected_scenario}")
                try:
                    st.video(str(video_path))
                except Exception:
                    st.warning("Direct MP4 playback was not available in this browser. The preview image and evidence snapshots remain available for inspection.")
                    preview_path = scenario_dir / "preview.webp"
                    if preview_path.exists():
                        st.image(str(preview_path), use_container_width=True)


if __name__ == "__main__":
    main()
