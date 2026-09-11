from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile
import html
import json
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as ui


st.set_page_config(
    page_title="TempSequence · LSTM v2",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
PREDICTIONS_NAME = "predictions_v2.csv"
KNOWN_FILES = {
    PREDICTIONS_NAME,
    "threshold_predictions_v2.csv",
    "experiment_results_v2.csv",
    "calibration_summary_v2.csv",
    "model_metadata_v2.json",
}
SUPPORTED_HORIZONS = (1, 2, 6)

PREDICTION_COLUMNS = {
    "as_of_timestamp",
    "target_timestamp",
    "horizon_hours",
    "actual_temperature_c",
    "predicted_temperature_c",
    "lower_10_c",
    "upper_90_c",
}
THRESHOLD_COLUMNS = {
    "as_of_timestamp",
    "target_timestamp",
    "horizon_hours",
    "strike_c",
    "actual_temperature_c",
    "predicted_temperature_c",
    "probability_yes",
    "observed_yes",
}
METRIC_COLUMNS = {
    "horizon_hours",
    "MAE",
    "RMSE",
    "persistence_MAE",
    "persistence_RMSE",
}
CALIBRATION_COLUMNS = {"probability_bin", "mean_probability", "observed_rate", "count"}


class BundleError(ValueError):
    """Raised when an uploaded or bundled artifact fails validation."""


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #070a0d;
            --panel: #10161c;
            --panel-2: #151d24;
            --border: #26323c;
            --mint: #26e0a5;
            --cyan: #72d9ff;
            --amber: #ffc857;
            --violet: #b49aff;
            --red: #ff7e7e;
            --muted: #8d9baa;
            --text: #edf4f7;
        }
        .stApp { background: var(--bg); color: var(--text); }
        [data-testid="stSidebar"] {
            background: #0b1015;
            border-right: 1px solid var(--border);
        }
        [data-testid="stHeader"] { background: rgba(7, 10, 13, .84); }
        .block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 4rem; }
        .brand { font-size: 1.35rem; font-weight: 800; letter-spacing: -.035em; }
        .brand span { color: var(--mint); }
        .eyebrow { color: var(--mint); font: 700 .68rem/1.2 ui-monospace, monospace; letter-spacing: .16em; }
        .hero-title { font-size: clamp(2rem, 4.5vw, 3.7rem); line-height: 1.02; margin: .4rem 0 .65rem; letter-spacing: -.055em; }
        .hero-copy { color: var(--muted); max-width: 780px; font-size: 1rem; line-height: 1.55; }
        .status-row { display: flex; flex-wrap: wrap; gap: .5rem; margin: 1rem 0 1.25rem; }
        .status-pill {
            display: inline-flex; align-items: center; gap: .35rem;
            border: 1px solid #82671d; color: var(--amber); background: #211b0b;
            padding: .28rem .62rem; border-radius: 999px;
            font: 700 .68rem/1.1 ui-monospace, monospace; letter-spacing: .055em;
        }
        .status-pill.good { border-color: #1b7e64; color: #67f0c2; background: #0b211b; }
        .status-pill.info { border-color: #236b86; color: var(--cyan); background: #0b1b24; }
        .section-kicker { color: var(--muted); font: 700 .68rem/1.2 ui-monospace, monospace; letter-spacing: .14em; margin: 1.25rem 0 .55rem; }
        .metric-card {
            background: linear-gradient(145deg, var(--panel-2), var(--panel));
            border: 1px solid var(--border); border-radius: 14px;
            padding: 1rem 1.05rem; min-height: 126px;
        }
        .metric-card.best { border-color: rgba(38, 224, 165, .68); box-shadow: inset 0 0 36px rgba(38, 224, 165, .045); }
        .metric-card.warn { border-color: rgba(255, 200, 87, .55); }
        .metric-label { color: var(--muted); font: 700 .68rem/1.2 ui-monospace, monospace; letter-spacing: .12em; }
        .metric-value { font-size: 1.75rem; font-weight: 800; margin: .5rem 0 .2rem; letter-spacing: -.035em; }
        .metric-foot { color: var(--muted); font-size: .76rem; line-height: 1.35; }
        .panel {
            background: linear-gradient(145deg, #131a21, #0e1419);
            border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.1rem;
        }
        .panel-title { font-size: 1rem; font-weight: 750; margin-bottom: .15rem; }
        .panel-subtitle { color: var(--muted); font-size: .78rem; margin-bottom: .7rem; }
        .truth-note { padding: .85rem 1rem; border-left: 3px solid var(--cyan); background: #0c171e; color: #bac8d1; border-radius: 0 9px 9px 0; margin: .8rem 0 1rem; font-size: .82rem; line-height: 1.45; }
        .danger-note { padding: .8rem 1rem; border-left: 3px solid var(--amber); background: #1c170c; color: #e9d9a9; border-radius: 0 9px 9px 0; font-size: .82rem; line-height: 1.45; }
        .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--border); }
        .stTabs [aria-selected="true"] { color: var(--mint) !important; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
        div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: .7rem .8rem; }
        .small-muted { color: var(--muted); font-size: .76rem; }
        .mini-stat-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .55rem; margin: .8rem 0 1rem; }
        .mini-stat { background: #0c1218; border: 1px solid var(--border); border-radius: 10px; padding: .62rem .72rem; min-height: 70px; }
        .mini-stat-label { color: var(--muted); font: 700 .61rem/1.2 ui-monospace, monospace; letter-spacing: .08em; text-transform: uppercase; }
        .mini-stat-value { color: var(--text); font-size: 1.08rem; font-weight: 750; margin: .3rem 0 .08rem; letter-spacing: -.025em; }
        .mini-stat-foot { color: var(--muted); font-size: .69rem; line-height: 1.25; }
        .contract-kicker { display: flex; align-items: baseline; justify-content: space-between; gap: .75rem; }
        .contract-chip { display: inline-flex; align-items: center; border: 1px solid rgba(255, 227, 215, .55); background: rgba(54, 16, 10, .35); color: #fff7f2; padding: .28rem .58rem; border-radius: 999px; font: 700 .67rem/1.1 ui-monospace, monospace; white-space: nowrap; }
        .control-label { color: var(--muted); font-size: .78rem; line-height: 1.3; height: 1.55rem; display: flex; align-items: flex-start; }
        .chart-legend { display: flex; flex-wrap: wrap; gap: .8rem 1.25rem; margin: .55rem 0 .8rem; color: var(--muted); font-size: .72rem; }
        .chart-legend-item { display: inline-flex; align-items: center; gap: .35rem; }
        .legend-swatch { width: .62rem; height: .62rem; border-radius: 2px; display: inline-block; }
        .legend-swatch.observed { background: #26e0a5; }
        .legend-swatch.other { background: #657583; }
        .legend-swatch.curve { width: .9rem; height: .16rem; border-radius: 999px; background: #72d9ff; }
        .legend-swatch.forecast { background: transparent; border: 2px solid #b49aff; }
        .signal-strip { display: flex; align-items: center; justify-content: space-between; gap: 1rem; background: linear-gradient(90deg, #111c25, #0c141b); border: 1px solid #315064; border-radius: 10px; padding: .7rem .85rem; margin: .75rem 0 .85rem; }
        .signal-label { color: var(--cyan); font: 700 .62rem/1.2 ui-monospace, monospace; letter-spacing: .1em; text-transform: uppercase; }
        .signal-value { color: var(--text); font-size: 1.08rem; font-weight: 750; }
        .signal-detail { color: var(--muted); font-size: .74rem; text-align: right; }
        .period-note { color: var(--muted); font-size: .72rem; line-height: 1.35; padding-top: .35rem; }
        @media (max-width: 800px) { .mini-stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise BundleError(f"{label} is missing required columns: {', '.join(missing)}")


def _parse_timestamps(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], errors="coerce")
        if result[column].isna().any():
            count = int(result[column].isna().sum())
            raise BundleError(f"{label} contains {count} invalid {column} value(s).")
    return result


def _parse_integer_column(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    result = frame.copy()
    values = pd.to_numeric(result[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise BundleError(f"{label} contains invalid {column} values.")
    if not np.allclose(values.to_numpy(), np.round(values.to_numpy())):
        raise BundleError(f"{label} contains non-integer {column} values.")
    result[column] = values.astype(int)
    return result


def _parse_numeric_columns(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        values = result[column].to_numpy(dtype=float)
        if result[column].isna().any() or not np.isfinite(values).all():
            raise BundleError(f"{label} contains invalid numeric values in {column}.")
    return result


def _parse_bool_column(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    result = frame.copy()
    if pd.api.types.is_bool_dtype(result[column]):
        return result
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    parsed = result[column].astype(str).str.strip().str.lower().map(mapping)
    if parsed.isna().any():
        raise BundleError(f"{label} contains invalid {column} values; use true/false.")
    result[column] = parsed.astype(bool)
    return result


def _validate_time_offsets(frame: pd.DataFrame, label: str) -> None:
    expected_hours = (frame["target_timestamp"] - frame["as_of_timestamp"]).dt.total_seconds() / 3600.0
    if not np.allclose(expected_hours.to_numpy(), frame["horizon_hours"].to_numpy(dtype=float)):
        raise BundleError(f"{label} contains rows whose target timestamp does not match horizon_hours.")


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, PREDICTION_COLUMNS, "predictions_v2.csv")
    result = _parse_timestamps(frame, ["as_of_timestamp", "target_timestamp"], "predictions_v2.csv")
    result = _parse_integer_column(result, "horizon_hours", "predictions_v2.csv")
    result = _parse_numeric_columns(
        result,
        ["actual_temperature_c", "predicted_temperature_c", "lower_10_c", "upper_90_c"],
        "predictions_v2.csv",
    )
    unsupported = sorted(set(result["horizon_hours"]) - set(SUPPORTED_HORIZONS))
    if unsupported:
        raise BundleError(f"predictions_v2.csv contains unsupported horizon(s): {unsupported}")
    if result.duplicated(["as_of_timestamp", "target_timestamp", "horizon_hours"]).any():
        raise BundleError("predictions_v2.csv contains duplicate forecast keys.")
    if (result["lower_10_c"] > result["upper_90_c"]).any():
        raise BundleError("predictions_v2.csv contains lower intervals above upper intervals.")
    _validate_time_offsets(result, "predictions_v2.csv")
    result = result.sort_values(["target_timestamp", "horizon_hours"]).reset_index(drop=True)
    if result.empty:
        raise BundleError("predictions_v2.csv contains no rows.")
    return result


def validate_thresholds(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, THRESHOLD_COLUMNS, "threshold_predictions_v2.csv")
    result = _parse_timestamps(frame, ["as_of_timestamp", "target_timestamp"], "threshold_predictions_v2.csv")
    result = _parse_integer_column(result, "horizon_hours", "threshold_predictions_v2.csv")
    result = _parse_numeric_columns(
        frame=result,
        columns=["strike_c", "actual_temperature_c", "predicted_temperature_c", "probability_yes"],
        label="threshold_predictions_v2.csv",
    )
    result = _parse_bool_column(result, "observed_yes", "threshold_predictions_v2.csv")
    if not set(result["horizon_hours"]).issubset(SUPPORTED_HORIZONS):
        raise BundleError("threshold_predictions_v2.csv contains unsupported horizons.")
    if ((result["probability_yes"] < 0) | (result["probability_yes"] > 1)).any():
        raise BundleError("threshold_predictions_v2.csv contains probabilities outside [0, 1].")
    if result.duplicated(["as_of_timestamp", "target_timestamp", "horizon_hours", "strike_c"]).any():
        raise BundleError("threshold_predictions_v2.csv contains duplicate threshold keys.")
    _validate_time_offsets(result, "threshold_predictions_v2.csv")
    return result.sort_values(["target_timestamp", "horizon_hours", "strike_c"]).reset_index(drop=True)


def validate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, METRIC_COLUMNS, "experiment_results_v2.csv")
    result = _parse_integer_column(frame, "horizon_hours", "experiment_results_v2.csv")
    numeric = [column for column in METRIC_COLUMNS if column != "horizon_hours"]
    result = _parse_numeric_columns(result, numeric, "experiment_results_v2.csv")
    if not set(result["horizon_hours"]).issubset(SUPPORTED_HORIZONS):
        raise BundleError("experiment_results_v2.csv contains unsupported horizons.")
    if result.duplicated("horizon_hours").any():
        raise BundleError("experiment_results_v2.csv must contain one row per horizon.")
    return result.sort_values("horizon_hours").reset_index(drop=True)


def validate_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, CALIBRATION_COLUMNS, "calibration_summary_v2.csv")
    result = _parse_numeric_columns(
        frame,
        ["mean_probability", "observed_rate", "count"],
        "calibration_summary_v2.csv",
    )
    if ((result["mean_probability"] < 0) | (result["mean_probability"] > 1)).any():
        raise BundleError("calibration_summary_v2.csv contains invalid mean probabilities.")
    if ((result["observed_rate"] < 0) | (result["observed_rate"] > 1)).any():
        raise BundleError("calibration_summary_v2.csv contains invalid observed rates.")
    if (result["count"] < 0).any():
        raise BundleError("calibration_summary_v2.csv contains negative counts.")
    return result.reset_index(drop=True)


def validate_metadata(value: object) -> dict:
    if not isinstance(value, dict):
        raise BundleError("model_metadata_v2.json must contain a JSON object.")
    return value


def parse_bundle(raw_files: dict[str, bytes]) -> dict[str, object]:
    if PREDICTIONS_NAME not in raw_files:
        raise BundleError("The bundle must include predictions_v2.csv.")

    bundle: dict[str, object] = {}
    try:
        bundle["predictions"] = validate_predictions(pd.read_csv(BytesIO(raw_files[PREDICTIONS_NAME])))
        if "threshold_predictions_v2.csv" in raw_files:
            bundle["thresholds"] = validate_thresholds(
                pd.read_csv(BytesIO(raw_files["threshold_predictions_v2.csv"]))
            )
        if "experiment_results_v2.csv" in raw_files:
            bundle["metrics"] = validate_metrics(
                pd.read_csv(BytesIO(raw_files["experiment_results_v2.csv"]))
            )
        if "calibration_summary_v2.csv" in raw_files:
            bundle["calibration"] = validate_calibration(
                pd.read_csv(BytesIO(raw_files["calibration_summary_v2.csv"]))
            )
        if "model_metadata_v2.json" in raw_files:
            bundle["metadata"] = validate_metadata(
                json.loads(raw_files["model_metadata_v2.json"].decode("utf-8"))
            )
    except (pd.errors.ParserError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"Could not parse the uploaded artifact: {error}") from error
    return bundle


def collect_uploaded_files(uploaded_files: list[object]) -> dict[str, bytes]:
    raw_files: dict[str, bytes] = {}
    for uploaded in uploaded_files:
        filename = Path(str(uploaded.name)).name
        if filename.lower().endswith(".zip"):
            try:
                with ZipFile(BytesIO(uploaded.getvalue())) as archive:
                    for member in archive.infolist():
                        member_name = Path(member.filename).name
                        if member.is_dir() or member_name not in KNOWN_FILES:
                            continue
                        if member_name in raw_files:
                            raise BundleError(f"The upload contains duplicate {member_name} files.")
                        raw_files[member_name] = archive.read(member)
            except BadZipFile as error:
                raise BundleError(f"{filename} is not a valid ZIP archive.") from error
        elif filename in KNOWN_FILES:
            if filename in raw_files:
                raise BundleError(f"The upload contains duplicate {filename} files.")
            raw_files[filename] = uploaded.getvalue()
        else:
            raise BundleError(
                f"Unsupported upload {filename!r}. Use the v2 CSV/JSON files or a ZIP bundle."
            )
    return raw_files


@st.cache_data(show_spinner=False)
def load_default_bundle() -> dict[str, object]:
    raw_files: dict[str, bytes] = {}
    for filename in KNOWN_FILES:
        path = ARTIFACT_DIR / filename
        if path.exists():
            raw_files[filename] = path.read_bytes()
    return parse_bundle(raw_files)


def metric_value(row: pd.Series | None, column: str) -> float | None:
    if row is None or column not in row or pd.isna(row[column]):
        return None
    return float(row[column])


def fmt_temp(value: float | None) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.2f}°C"


def compact_stats(items: list[tuple[str, str, str]]) -> None:
    """Render small inline context stats below the primary panel."""
    cards = []
    for label, value, foot in items:
        cards.append(
            f'<div class="mini-stat"><div class="mini-stat-label">{html.escape(label)}</div>'
            f'<div class="mini-stat-value">{html.escape(value)}</div>'
            f'<div class="mini-stat-foot">{html.escape(foot)}</div></div>'
        )
    st.markdown(f'<div class="mini-stat-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def benchmark_rows(predictions: pd.DataFrame, metrics: pd.DataFrame | None) -> pd.DataFrame:
    if metrics is not None:
        return metrics.copy()
    rows = []
    for horizon in SUPPORTED_HORIZONS:
        subset = predictions[predictions["horizon_hours"] == horizon]
        if subset.empty:
            continue
        error = subset["actual_temperature_c"] - subset["predicted_temperature_c"]
        rows.append(
            {
                "horizon_hours": horizon,
                "MAE": float(error.abs().mean()),
                "RMSE": float(np.sqrt(np.mean(np.square(error)))),
                "persistence_MAE": np.nan,
                "persistence_RMSE": np.nan,
            }
        )
    return pd.DataFrame(rows)


def horizon_row(metrics: pd.DataFrame, horizon: int) -> pd.Series | None:
    rows = metrics.loc[metrics["horizon_hours"] == horizon]
    return None if rows.empty else rows.iloc[0]


def forecast_chart(frame: pd.DataFrame, horizon: int) -> go.Figure:
    figure = go.Figure()
    x = frame["target_timestamp"]
    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame["upper_90_c"],
            mode="lines",
            line={"width": 0, "color": "rgba(114, 217, 255, 0)"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame["lower_10_c"],
            mode="lines",
            line={"width": 0, "color": "rgba(114, 217, 255, 0)"},
            fill="tonexty",
            fillcolor="rgba(114, 217, 255, .12)",
            name="10–90% interval",
            hovertemplate="%{x|%d %b %Y · %H:%M}<br>Interval: %{y:.2f}°C<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame["actual_temperature_c"],
            mode="lines+markers",
            name="Observed",
            line={"color": "#26e0a5", "width": 2.8},
            marker={"size": 4, "color": "#26e0a5"},
            hovertemplate="%{x|%d %b %Y · %H:%M}<br>Observed: %{y:.2f}°C<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame["predicted_temperature_c"],
            mode="lines+markers",
            name=f"LSTM · +{horizon}h",
            line={"color": "#72d9ff", "width": 2.1, "dash": "dot"},
            marker={"size": 3, "color": "#72d9ff"},
            hovertemplate="%{x|%d %b %Y · %H:%M}<br>Forecast: %{y:.2f}°C<extra></extra>",
        )
    )
    figure.update_layout(
        height=480,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="#131a21",
        plot_bgcolor="#131a21",
        font={"color": "#aab8c4", "family": "Inter, sans-serif"},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.17, "x": 0.5, "xanchor": "center"},
        xaxis={"gridcolor": "#232d36", "showline": False},
        yaxis={"title": "Temperature (°C)", "gridcolor": "#232d36", "zeroline": False},
    )
    return figure


def bracket_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert calibrated cumulative thresholds into exclusive temperature brackets."""
    ordered = frame.sort_values("strike_c").copy()
    strikes = ordered["strike_c"].to_numpy(dtype=float)
    cumulative = np.clip(ordered["probability_yes"].to_numpy(dtype=float), 0.0, 1.0)
    # Threshold probabilities should be non-increasing. Clip small calibration
    # inversions so bracket masses remain valid probabilities.
    cumulative = np.minimum.accumulate(cumulative)
    masses = np.concatenate(([1.0 - cumulative[0]], -np.diff(cumulative), [cumulative[-1]]))
    masses = np.clip(masses, 0.0, None)
    total = float(masses.sum())
    if total <= 0:
        masses = np.full(len(masses), 1.0 / len(masses))
    else:
        masses = masses / total

    labels = [f"< {strikes[0]:.0f}°C"]
    lower_bounds: list[float | None] = [None]
    upper_bounds: list[float | None] = [float(strikes[0])]
    for left, right in zip(strikes[:-1], strikes[1:]):
        labels.append(f"{left:.0f}–<{right:.0f}°C")
        lower_bounds.append(float(left))
        upper_bounds.append(float(right))
    labels.append(f"≥ {strikes[-1]:.0f}°C")
    lower_bounds.append(float(strikes[-1]))
    upper_bounds.append(None)

    actual = float(ordered["actual_temperature_c"].iloc[0])
    predicted = float(ordered["predicted_temperature_c"].iloc[0])
    observed_bucket = []
    predicted_bucket = []
    for lower, upper in zip(lower_bounds, upper_bounds):
        observed_bucket.append(
            (lower is None and actual < float(upper))
            or (upper is None and actual >= float(lower))
            or (lower is not None and upper is not None and lower <= actual < upper)
        )
        predicted_bucket.append(
            (lower is None and predicted < float(upper))
            or (upper is None and predicted >= float(lower))
            or (lower is not None and upper is not None and lower <= predicted < upper)
        )
    return pd.DataFrame(
        {
            "bracket_label": labels,
            "lower_bound_c": lower_bounds,
            "upper_bound_c": upper_bounds,
            "probability_yes": masses,
            "observed_yes": observed_bucket,
            "predicted_bracket": predicted_bucket,
            "actual_temperature_c": actual,
            "predicted_temperature_c": predicted,
        }
    )


def bracket_chart(frame: pd.DataFrame) -> go.Figure:
    ordered = frame.copy()
    labels = ordered["bracket_label"].tolist()
    colors = ["#26e0a5" if bool(value) else "#657583" for value in ordered["observed_yes"]]
    line_colors = ["#b49aff" if bool(value) else "#27333c" for value in ordered["predicted_bracket"]]
    line_widths = [2.0 if bool(value) else 0.8 for value in ordered["predicted_bracket"]]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=ordered["probability_yes"] * 100,
            marker_color=colors,
            marker_line_color=line_colors,
            marker_line_width=line_widths,
            text=[f"{value * 100:.1f}%" for value in ordered["probability_yes"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Bracket %{x}<br>Probability YES: %{y:.1f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=labels,
            y=ordered["probability_yes"] * 100,
            mode="lines+markers",
            line={"color": "#72d9ff", "width": 2.2},
            marker={"color": "#72d9ff", "size": 6, "line": {"color": "#131a21", "width": 1}},
            text=[f"{value * 100:.1f}%" for value in ordered["probability_yes"]],
            textposition="top center",
            textfont={"color": "#aab8c4", "size": 11},
            name="Probability",
            hovertemplate="Bracket %{x}<br>Probability YES: %{y:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        height=380,
        margin={"l": 16, "r": 16, "t": 46, "b": 24},
        paper_bgcolor="#131a21",
        plot_bgcolor="#131a21",
        font={"color": "#aab8c4", "family": "Inter, sans-serif"},
        yaxis={"title": "Synthetic probability (%)", "range": [0, 108], "gridcolor": "#27333c", "zeroline": False},
        xaxis={"title": "Temperature bracket", "gridcolor": "#27333c", "linecolor": "#27333c"},
        showlegend=False,
    )
    predicted_index = int(np.flatnonzero(ordered["predicted_bracket"].to_numpy(dtype=bool))[0])
    observed_index = int(np.flatnonzero(ordered["observed_yes"].to_numpy(dtype=bool))[0])
    figure.add_annotation(
        x=labels[predicted_index], y=1.04, xref="x", yref="paper",
        text="FORECAST", showarrow=False, font={"color": "#b49aff", "size": 10},
    )
    if observed_index != predicted_index:
        figure.add_annotation(
            x=labels[observed_index], y=0.96, xref="x", yref="paper",
            text="OBSERVED", showarrow=False, font={"color": "#26e0a5", "size": 10},
        )
    return figure


def calibration_chart(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
            line={"color": "#657583", "dash": "dash"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["mean_probability"], y=frame["observed_rate"],
            mode="lines+markers", name="Observed", line={"color": "#26e0a5", "width": 2.5},
            marker={"size": 8, "color": "#72d9ff"},
            customdata=frame[["count"]],
            hovertemplate="Predicted: %{x:.1%}<br>Observed: %{y:.1%}<br>Rows: %{customdata[0]:,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=390,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="#131a21", plot_bgcolor="#131a21",
        font={"color": "#aab8c4", "family": "Inter, sans-serif"},
        xaxis={"title": "Mean predicted probability", "range": [0, 1], "tickformat": ".0%", "gridcolor": "#27333c"},
        yaxis={"title": "Observed frequency", "range": [0, 1], "tickformat": ".0%", "gridcolor": "#27333c"},
        legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"},
    )
    return figure


def display_prediction_table(frame: pd.DataFrame) -> None:
    table = frame.sort_values("target_timestamp", ascending=False).head(120).copy()
    table["absolute_error_c"] = (table["predicted_temperature_c"] - table["actual_temperature_c"]).abs()
    table = table[
        [
            "as_of_timestamp", "target_timestamp", "actual_temperature_c",
            "predicted_temperature_c", "lower_10_c", "upper_90_c", "absolute_error_c",
        ]
    ].rename(
        columns={
            "as_of_timestamp": "As of",
            "target_timestamp": "Target",
            "actual_temperature_c": "Observed (°C)",
            "predicted_temperature_c": "Forecast (°C)",
            "lower_10_c": "P10 (°C)",
            "upper_90_c": "P90 (°C)",
            "absolute_error_c": "Abs. error (°C)",
        }
    )
    st.dataframe(
        table.style.format(
            {
                "Observed (°C)": "{:.2f}", "Forecast (°C)": "{:.2f}",
                "P10 (°C)": "{:.2f}", "P90 (°C)": "{:.2f}", "Abs. error (°C)": "{:.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
        height=360,
    )


def main() -> None:
    inject_styles()

    with st.sidebar:
        st.markdown('<div class="brand">🌡️ Temp<span>Sequence</span></div>', unsafe_allow_html=True)
        st.caption("LSTM v2 · short-horizon weather lab")
        st.divider()
        uploads = st.file_uploader(
            "Upload v2 export bundle",
            type=["csv", "json", "zip"],
            accept_multiple_files=True,
            help="Upload predictions_v2.csv alone, optional companion exports, or a ZIP bundle.",
        )
        if uploads:
            try:
                bundle = parse_bundle(collect_uploaded_files(uploads))
                source_label = "UPLOADED BUNDLE"
            except BundleError as error:
                st.error(str(error))
                st.stop()
        else:
            try:
                bundle = load_default_bundle()
                source_label = "BUNDLED V2 EXPORT"
            except BundleError as error:
                st.error(f"Bundled artifacts are unavailable: {error}")
                st.info("Upload predictions_v2.csv or a v2 ZIP bundle to continue.")
                st.stop()

        predictions = bundle["predictions"]
        assert isinstance(predictions, pd.DataFrame)
        available_horizons = [horizon for horizon in SUPPORTED_HORIZONS if horizon in set(predictions["horizon_hours"])]
        if not available_horizons:
            st.error("The prediction file has no supported horizons (+1h, +2h, +6h).")
            st.stop()
        default_horizon = 1 if 1 in available_horizons else available_horizons[0]
        horizon = ui.select(
            "Primary forecast horizon",
            options=available_horizons,
            value=default_horizon,
            format_func=lambda value: f"+{value} hour" if value == 1 else f"+{value} hours",
            key="primary_horizon",
        )
        if horizon is None:
            horizon = default_horizon
        horizon_frame = predictions[predictions["horizon_hours"] == horizon]
        max_window = max(24, min(24 * 30, len(horizon_frame)))
        default_window = min(24 * 7, max_window)
        visible_window = ui.slider(
            "Visible target hours",
            min_value=24,
            max_value=max_window,
            value=default_window,
            step=24,
            key="visible_target_hours",
        )
        st.divider()
        st.markdown("**Data boundary**")
        st.caption("2016 chronological test period\n\nHourly mean temperature\n\n72-hour input history")
        ui.badge(source_label, variant="outline", key="sidebar_source")

    metrics_file = bundle.get("metrics")
    calibration = bundle.get("calibration")
    thresholds = bundle.get("thresholds")
    metadata = bundle.get("metadata") or {}
    assert metrics_file is None or isinstance(metrics_file, pd.DataFrame)
    assert calibration is None or isinstance(calibration, pd.DataFrame)
    assert thresholds is None or isinstance(thresholds, pd.DataFrame)
    assert isinstance(metadata, dict)

    metrics = benchmark_rows(predictions, metrics_file)
    selected = horizon_frame.tail(visible_window).reset_index(drop=True)
    latest = selected.iloc[-1]
    latest_error = abs(float(latest["predicted_temperature_c"]) - float(latest["actual_temperature_c"]))
    selected_mae = float(np.mean(np.abs(selected["predicted_temperature_c"] - selected["actual_temperature_c"])))

    st.markdown('<div class="eyebrow">SEQUENTIAL WEATHER LAB · LSTM V2</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Short-horizon temperature intelligence</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">A 72-hour LSTM backtest with direct +1h, +2h, and +6h forecasts. The primary view follows the short horizons while preserving transparent benchmark and calibration evidence.</div>',
        unsafe_allow_html=True,
    )
    active_tab = ui.tabs(
        ["contracts", "monitor", "calibration", "benchmark"],
        value="contracts",
        format_func=lambda value: {
            "monitor": "Forecast monitor",
            "contracts": "Temperature brackets",
            "calibration": "Calibration",
            "benchmark": "Benchmark & model",
        }[value],
        key="dashboard_tabs",
        label="Dashboard sections",
        variant="line",
        width="stretch",
    )

    if active_tab == "monitor":
        chart_left, chart_right = st.columns([2.25, 1])
        with chart_left:
            st.markdown('<div class="panel-title">Observed versus forecast</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="panel-subtitle">Target timestamps · selected +{horizon}h horizon · last {len(selected):,} rows</div>', unsafe_allow_html=True)
            st.plotly_chart(forecast_chart(selected, horizon), width="stretch", config={"displayModeBar": False})
        with chart_right:
            st.markdown('<div class="panel-title">Latest forecast record</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-subtitle">One row from the chronological test export</div>', unsafe_allow_html=True)
            st.metric("Target time", pd.Timestamp(latest["target_timestamp"]).strftime("%d %b %Y · %H:%M"))
            st.metric("Forecast", f"{float(latest['predicted_temperature_c']):.2f}°C")
            st.metric("Observed", f"{float(latest['actual_temperature_c']):.2f}°C")
            st.metric("Absolute error", f"{latest_error:.2f}°C")
            st.download_button(
                "Download filtered forecasts",
                data=selected.to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_v2_plus_{horizon}h_filtered.csv",
                mime="text/csv",
                width="stretch",
            )
        st.markdown('<div class="section-kicker">FORECAST RECORDS</div>', unsafe_allow_html=True)
        display_prediction_table(selected)

    elif active_tab == "contracts":
        if thresholds is None:
            st.info("No threshold_predictions_v2.csv was included in this upload. Upload the companion file or a complete ZIP bundle to enable this panel.")
        else:
            threshold_targets = sorted(
                set(selected["target_timestamp"]).intersection(
                    thresholds.loc[thresholds["horizon_hours"] == horizon, "target_timestamp"]
                )
            )
            all_threshold_targets = sorted(
                set(thresholds.loc[thresholds["horizon_hours"] == horizon, "target_timestamp"])
            )
            if not threshold_targets:
                st.info("No threshold rows match the selected horizon and visible forecast window.")
            else:
                stored_target = st.session_state.get("contract_target_default", threshold_targets[-1])
                if stored_target not in threshold_targets:
                    target_options = sorted(set(threshold_targets + [stored_target]))
                else:
                    target_options = threshold_targets
                target_control, random_control = st.columns([4, 1])
                with target_control:
                    target_choice = ui.select(
                        "Contract target timestamp",
                        options=target_options,
                        value=stored_target,
                        format_func=lambda value: pd.Timestamp(value).strftime("%d %b %Y · %H:%M"),
                        key="contract_target_timestamp",
                    )
                if target_choice is None:
                    target_choice = stored_target
                st.session_state["contract_target_default"] = target_choice
                with random_control:
                    st.markdown('<div class="control-label">Explore history</div>', unsafe_allow_html=True)
                    if ui.button("↻ Randomize", variant="secondary", size="sm", key="randomize_contract_timestamp", width="stretch"):
                        candidates = [value for value in all_threshold_targets if value != target_choice]
                        st.session_state["contract_target_default"] = random.choice(candidates or all_threshold_targets)
                        st.rerun()
                target_index = next(
                    (index for index, value in enumerate(all_threshold_targets) if pd.Timestamp(value) == pd.Timestamp(target_choice)),
                    0,
                )
                navigation_left, navigation_right, period_note = st.columns([1, 1, 4])
                with navigation_left:
                    if ui.button("← Previous", variant="outline", size="sm", key="previous_contract_timestamp", width="stretch"):
                        st.session_state["contract_target_default"] = all_threshold_targets[(target_index - 1) % len(all_threshold_targets)]
                        st.rerun()
                with navigation_right:
                    if ui.button("Next →", variant="outline", size="sm", key="next_contract_timestamp", width="stretch"):
                        st.session_state["contract_target_default"] = all_threshold_targets[(target_index + 1) % len(all_threshold_targets)]
                        st.rerun()
                with period_note:
                    full_start = pd.Timestamp(all_threshold_targets[0]).strftime("%d %b %Y · %H:%M")
                    full_end = pd.Timestamp(all_threshold_targets[-1]).strftime("%d %b %Y · %H:%M")
                    visible_start = pd.Timestamp(threshold_targets[0]).strftime("%d %b %Y")
                    visible_end = pd.Timestamp(threshold_targets[-1]).strftime("%d %b %Y")
                    outside_window = target_choice not in threshold_targets
                    outside_note = " · selected outside visible window" if outside_window else ""
                    st.markdown(
                        f'<div class="period-note">Full test period: {full_start} → {full_end}<br>Visible window: {visible_start} → {visible_end}{outside_note}</div>',
                        unsafe_allow_html=True,
                    )
                contract_rows = thresholds[
                    (thresholds["horizon_hours"] == horizon)
                    & (thresholds["target_timestamp"] == target_choice)
                ].sort_values("strike_c")
                bracket_rows = bracket_distribution(contract_rows)
                st.markdown(
                    f'<div class="contract-kicker"><div><div class="panel-title">Temperature bracket probabilities · +{horizon}h</div><div class="panel-subtitle">Mutually exclusive temperature ranges · calibrated bracket probabilities</div></div><span class="contract-chip">{pd.Timestamp(target_choice).strftime("%d %b · %H:%M")}</span></div>',
                    unsafe_allow_html=True,
                )
                most_likely_index = int(bracket_rows["probability_yes"].idxmax())
                most_likely = bracket_rows.loc[most_likely_index]
                observed_label = "YES" if bool(most_likely["observed_yes"]) else "NO"
                st.markdown(
                    f'<div class="signal-strip"><div><div class="signal-label">Most likely bracket</div><div class="signal-value">{html.escape(str(most_likely["bracket_label"]))} · {float(most_likely["probability_yes"]):.1%}</div></div><div class="signal-detail">Observed outcome for this bracket: <strong>{observed_label}</strong><br>Probabilities sum to 100%</div></div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(bracket_chart(bracket_rows), width="stretch", config={"displayModeBar": False})
                st.markdown(
                    '<div class="chart-legend"><span class="chart-legend-item"><span class="legend-swatch observed"></span>Observed bracket (YES)</span><span class="chart-legend-item"><span class="legend-swatch other"></span>Other bracket probabilities</span><span class="chart-legend-item"><span class="legend-swatch forecast"></span>Forecast bracket</span><span class="chart-legend-item"><span class="legend-swatch curve"></span>Calibrated probability curve</span></div>',
                    unsafe_allow_html=True,
                )
                table = bracket_rows[["bracket_label", "probability_yes", "observed_yes"]].copy()
                table["probability_yes"] = table["probability_yes"].map(lambda value: f"{value * 100:.1f}%")
                table["observed_yes"] = table["observed_yes"].map(lambda value: "YES" if value else "NO")
                table.columns = ["Temperature bracket", "Probability YES", "Observed outcome"]
                st.dataframe(table, hide_index=True, width="stretch")
                record = bracket_rows.iloc[0]
                compact_stats(
                    [
                        ("Forecast temperature", f"{float(record['predicted_temperature_c']):.2f}°C", "model point estimate"),
                        ("Actual temperature", f"{float(record['actual_temperature_c']):.2f}°C", "observed target value"),
                        ("Visible-window MAE", f"{selected_mae:.2f}°C", f"+{horizon}h · last {len(selected):,} rows"),
                        ("Point error", f"{float(record['predicted_temperature_c']) - float(record['actual_temperature_c']):+.2f}°C", "forecast minus observed"),
                    ]
                )
                st.markdown('<div class="danger-note"><strong>Synthetic only:</strong> each row is a binary bracket contract. Probabilities are derived from calibrated cumulative thresholds and are not official Kalshi market prices or probabilities.</div>', unsafe_allow_html=True)

    elif active_tab == "calibration":
        if calibration is None:
            st.info("No calibration_summary_v2.csv was included in this upload.")
        else:
            left, right = st.columns([1.7, 1])
            with left:
                st.markdown('<div class="panel-title">Reliability curve</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-subtitle">Synthetic threshold probabilities versus observed event frequency</div>', unsafe_allow_html=True)
                st.plotly_chart(calibration_chart(calibration), width="stretch", config={"displayModeBar": False})
            with right:
                st.markdown('<div class="panel-title">Calibration bins</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-subtitle">Aggregated across horizons and strikes</div>', unsafe_allow_html=True)
                calibration_table = calibration.copy()
                calibration_table["mean_probability"] = calibration_table["mean_probability"].map(lambda value: f"{value:.1%}")
                calibration_table["observed_rate"] = calibration_table["observed_rate"].map(lambda value: f"{value:.1%}")
                calibration_table["count"] = calibration_table["count"].map(lambda value: f"{value:,.0f}")
                st.dataframe(calibration_table, hide_index=True, width="stretch", height=390)
            st.markdown('<div class="truth-note">Calibration is evaluated on the synthetic threshold contracts generated from validation residuals. Good reliability does not make these official market probabilities.</div>', unsafe_allow_html=True)

    elif active_tab == "benchmark":
        if metrics_file is None:
            st.info("No experiment_results_v2.csv was included; metrics are computed from the prediction rows and persistence comparisons are unavailable.")
        else:
            st.markdown('<div class="panel-title">Chronological test benchmark</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-subtitle">Metrics are read from experiment_results_v2.csv and remain separate from the visible-window metrics above.</div>', unsafe_allow_html=True)
            display_metrics = metrics.copy()
            display_metrics["Horizon"] = display_metrics["horizon_hours"].map(lambda value: f"+{value}h")
            display_columns = ["Horizon", "MAE", "RMSE", "persistence_MAE", "persistence_RMSE"]
            optional_columns = [column for column in ["v1_reference_MAE", "v1_reference_RMSE", "brier_score"] if column in display_metrics.columns]
            display_columns += optional_columns
            display_metrics = display_metrics[display_columns].rename(
                columns={
                    "MAE": "Model MAE", "RMSE": "Model RMSE",
                    "persistence_MAE": "Persistence MAE", "persistence_RMSE": "Persistence RMSE",
                    "v1_reference_MAE": "v1 +6h MAE", "v1_reference_RMSE": "v1 +6h RMSE",
                    "brier_score": "Brier score",
                }
            )
            st.dataframe(
                display_metrics.style.format({column: "{:.3f}" for column in display_metrics.columns if column != "Horizon"}),
                hide_index=True,
                width="stretch",
            )
            selected_metric = horizon_row(metrics, horizon)
            selected_mae_metric = metric_value(selected_metric, "MAE")
            selected_rmse_metric = metric_value(selected_metric, "RMSE")
            selected_brier = metric_value(selected_metric, "brier_score")
            calibration_gap = None
            if calibration is not None and not calibration.empty:
                calibration_gap = float(
                    np.mean(np.abs(calibration["mean_probability"] - calibration["observed_rate"]))
                )
            st.markdown(f'<div class="section-kicker">+{horizon}H QUALITY SUMMARY</div>', unsafe_allow_html=True)
            compact_stats(
                [
                    ("Test forecasts", f"{len(horizon_frame):,}", "chronological test rows"),
                    ("MAE / RMSE", f"{fmt_temp(selected_mae_metric)} / {fmt_temp(selected_rmse_metric)}", "selected horizon"),
                    ("Brier score", "—" if selected_brier is None else f"{selected_brier:.3f}", "synthetic bracket contracts"),
                    ("Calibration gap", "—" if calibration_gap is None else f"{calibration_gap:.1%}", "mean absolute bin gap"),
                ]
            )
            six_hour = horizon_row(metrics, 6)
            if six_hour is not None and metric_value(six_hour, "v1_reference_MAE") is not None:
                v1_mae = metric_value(six_hour, "v1_reference_MAE")
                v1_rmse = metric_value(six_hour, "v1_reference_RMSE")
                v2_mae = metric_value(six_hour, "MAE")
                v2_rmse = metric_value(six_hour, "RMSE")
                gate_passed = all(value is not None for value in [v1_mae, v1_rmse, v2_mae, v2_rmse]) and v2_mae < v1_mae and v2_rmse < v1_rmse
                if gate_passed:
                    st.success("The exported +6h result beats the recorded v1 reference on both MAE and RMSE.")
                else:
                    st.warning("The exported +6h result does not beat the recorded v1 reference on both metrics.")

        st.markdown('<div class="section-kicker">MODEL METADATA</div>', unsafe_allow_html=True)
        if metadata:
            summary = {
                "Model": metadata.get("model_name", "—"),
                "Dataset": metadata.get("dataset", "—"),
                "History": f"{metadata.get('history_hours', '—')} hours",
                "Horizons": ", ".join(f"+{value}h" for value in metadata.get("horizons_hours", [])),
                "Primary horizons": ", ".join(f"+{value}h" for value in metadata.get("primary_horizons_hours", [])),
                "Parameters": f"{metadata.get('parameters', 0):,}" if isinstance(metadata.get("parameters"), (int, float)) else "—",
                "Completed epochs": metadata.get("epochs_completed", "—"),
                "Training timestamp": metadata.get("training_timestamp_utc", "—"),
            }
            st.dataframe(pd.DataFrame([summary]), hide_index=True, width="stretch")
            with st.expander("Feature order and calibration details"):
                st.json(
                    {
                        "feature_columns": metadata.get("feature_columns", []),
                        "model_inputs": metadata.get("model_inputs", []),
                        "horizon_loss_weights": metadata.get("horizon_loss_weights", []),
                        "calibration_method": metadata.get("calibration_method", "—"),
                        "synthetic_contracts": metadata.get("synthetic_contracts", True),
                    }
                )
        else:
            st.info("No model_metadata_v2.json was included in this upload.")

        st.markdown('<div class="truth-note"><strong>Deployment boundary:</strong> this app visualizes precomputed artifact exports. It does not load TensorFlow, call a weather API, or produce a live forecast. A live version would require a current 72-hour sequence with the exact 19-feature preprocessing pipeline.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
