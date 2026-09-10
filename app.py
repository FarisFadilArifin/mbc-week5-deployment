from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="TempSequence — LSTM",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISPLAY_COLUMNS = [
    "timestamp",
    "actual_temperature",
    "lstm_prediction",
]
REQUIRED_COLUMNS = set(DISPLAY_COLUMNS)
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_PREDICTIONS = ARTIFACT_DIR / "predictions_v1.csv"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --mint: #26e0a5;
            --cyan: #69d2ff;
            --violet: #ad8cff;
            --muted: #8c98aa;
            --panel: #12171d;
            --border: #26303a;
        }
        .stApp { background: #080b0e; color: #eef3f8; }
        [data-testid="stSidebar"] { background: #0d1116; border-right: 1px solid var(--border); }
        [data-testid="stHeader"] { background: rgba(8, 11, 14, .85); }
        .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 4rem; }
        .brand { font-size: 1.25rem; font-weight: 750; letter-spacing: -.02em; }
        .brand span { color: var(--mint); }
        .eyebrow { color: var(--mint); font: 700 .72rem/1.2 monospace; letter-spacing: .14em; }
        .hero-title { font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1.04; margin: .45rem 0; letter-spacing: -.045em; }
        .hero-copy { color: var(--muted); max-width: 760px; font-size: 1rem; }
        .metric-card {
            background: linear-gradient(145deg, #12181e, #0e1318);
            border: 1px solid var(--border); border-radius: 12px;
            padding: 1rem 1.05rem; min-height: 126px;
        }
        .metric-card.best { border-color: rgba(38, 224, 165, .65); box-shadow: inset 0 0 30px rgba(38,224,165,.04); }
        .metric-label { color: var(--muted); font: 700 .7rem/1.2 monospace; letter-spacing: .12em; }
        .metric-value { font-size: 1.85rem; font-weight: 760; margin: .45rem 0 .2rem; }
        .metric-foot { color: var(--muted); font-size: .78rem; }
        .status-pill { display: inline-block; border: 1px solid #8c6d1f; color: #ffd267; background: #211b0b;
            padding: .26rem .58rem; border-radius: 999px; font: 700 .7rem/1 monospace; letter-spacing: .05em; }
        .truth-note { padding: .85rem 1rem; border-left: 3px solid var(--cyan); background: #0d151b;
            color: #b9c5d2; border-radius: 0 8px 8px 0; margin: 1rem 0; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
        .stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid var(--border); }
        .stTabs [aria-selected="true"] { color: var(--mint) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def preview_data() -> pd.DataFrame:
    """Illustrative data for UI review only; never presented as model output."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2016-06-18 00:00", periods=96, freq="h")
    hour = np.arange(len(timestamps))
    actual = 14 + 6 * np.sin((hour - 7) * 2 * np.pi / 24) + rng.normal(0, 0.45, len(hour))
    lstm = pd.Series(actual).rolling(4, min_periods=1).mean().shift(1).bfill().to_numpy() + 0.35
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual_temperature": actual,
            "lstm_prediction": lstm,
        }
    )


def parse_csv(file: BytesIO) -> pd.DataFrame:
    frame = pd.read_csv(file)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    frame = frame[DISPLAY_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for column in REQUIRED_COLUMNS - {"timestamp"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().drop_duplicates("timestamp").sort_values("timestamp")
    if frame.empty:
        raise ValueError("The CSV has no valid rows after parsing.")
    return frame


@st.cache_data
def load_prediction_file(path: str) -> pd.DataFrame:
    return parse_csv(path)


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def metric_card(label: str, value: str, foot: str, best: bool = False) -> None:
    class_name = "metric-card best" if best else "metric-card"
    st.markdown(
        f"""
        <div class="{class_name}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def forecast_chart(frame: pd.DataFrame) -> go.Figure:
    colors = {"Observed": "#26e0a5", "LSTM · A · 72h": "#69d2ff"}
    series = {
        "Observed": ("actual_temperature", 3.2, None),
        "LSTM · A · 72h": ("lstm_prediction", 2.0, "dot"),
    }
    figure = go.Figure()
    for label, (column, width, dash) in series.items():
        figure.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame[column],
                name=label,
                mode="lines",
                line={"color": colors[label], "width": width, "dash": dash},
                hovertemplate=f"%{{x|%d %b, %H:%M}}<br>{label}: %{{y:.1f}}°C<extra></extra>",
            )
        )
    figure.update_layout(
        height=470,
        margin={"l": 10, "r": 10, "t": 25, "b": 10},
        paper_bgcolor="#12171d",
        plot_bgcolor="#12171d",
        font={"color": "#aab5c2", "family": "Inter, sans-serif"},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"},
        xaxis={"gridcolor": "#232b33", "showline": False},
        yaxis={"title": "Temperature (°C)", "gridcolor": "#232b33", "zeroline": False},
    )
    return figure


inject_styles()

with st.sidebar:
    st.markdown('<div class="brand">🌡️ Temp<span>Sequence</span></div>', unsafe_allow_html=True)
    st.caption("Jena LSTM dashboard")
    st.divider()
    uploaded = st.file_uploader(
        "Kaggle prediction export",
        type="csv",
        help="Expected columns: timestamp, actual_temperature, lstm_prediction",
    )
    window = st.slider("Visible observations", 24, 336, 96, step=24)
    st.divider()
    if DEFAULT_PREDICTIONS.exists():
        st.success("Kaggle LSTM predictions loaded")
        st.caption("The dashboard is using the exported LSTM prediction file in the artifacts folder.")
    st.markdown("**Forecast definition**")
    st.caption("Target: hourly mean temperature\n\nHorizon: +6 hours\n\nDataset: Jena Climate 2009–2016")
    st.markdown("[Open Kaggle notebook](https://www.kaggle.com/code/farisfadilarifin/lstm-gru-predict-temperature)")

is_preview = uploaded is None and not DEFAULT_PREDICTIONS.exists()
if uploaded is not None:
    try:
        data = parse_csv(uploaded)
    except ValueError as error:
        st.error(str(error))
        st.stop()
elif DEFAULT_PREDICTIONS.exists():
    try:
        data = load_prediction_file(str(DEFAULT_PREDICTIONS))
    except ValueError as error:
        st.error(f"Could not load the bundled Kaggle export: {error}")
        st.stop()
else:
    data = preview_data()

data = data[DISPLAY_COLUMNS].tail(window).reset_index(drop=True)
actual = data["actual_temperature"]
lstm_mae = mae(actual, data["lstm_prediction"])
latest = data.iloc[-1]

st.markdown('<div class="eyebrow">SEQUENTIAL WEATHER LAB</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">LSTM versus reality</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">A focused LSTM dashboard forecasting hourly mean temperature six hours ahead.</div>',
    unsafe_allow_html=True,
)
if is_preview:
    st.markdown('<p><span class="status-pill">ILLUSTRATIVE PREVIEW — UPLOAD KAGGLE CSV</span></p>', unsafe_allow_html=True)
else:
    st.markdown('<p><span class="status-pill" style="border-color:#1c7f68;color:#65f1bf;background:#0b211b;">KAGGLE LSTM EXPORT LOADED</span></p>', unsafe_allow_html=True)

cols = st.columns(4)
with cols[0]:
    metric_card("OBSERVED", f"{latest['actual_temperature']:.1f}°C", data["timestamp"].max().strftime("%d %b %Y · %H:%M"))
with cols[1]:
    metric_card("LSTM · A · 72H", f"{latest['lstm_prediction']:.1f}°C", f"Window MAE {lstm_mae:.2f}°C", True)
with cols[2]:
    metric_card("FORECAST HORIZON", "+6h", "Hourly mean temperature")
with cols[3]:
    metric_card("HISTORY WINDOW", "72h", "Sequential input")

overview, performance, methodology = st.tabs(["Forecast timeline", "Performance", "Methodology"])

with overview:
    st.plotly_chart(forecast_chart(data), width="stretch", config={"displayModeBar": False})
    table = data.sort_values("timestamp", ascending=False).copy()
    table["LSTM error"] = (table["lstm_prediction"] - table["actual_temperature"]).abs()
    table.columns = ["Time", "Observed", "LSTM", "LSTM error"]
    st.dataframe(
        table.style.format({column: "{:.2f}°C" for column in table.columns if column != "Time"}),
        hide_index=True,
        width="stretch",
        height=320,
    )

with performance:
    left, right = st.columns(2)
    with left:
        st.subheader("LSTM error on visible window")
        metrics = pd.DataFrame(
            {
                "Model": ["LSTM · A · 72h"],
                "MAE": [lstm_mae],
                "RMSE": [rmse(actual, data["lstm_prediction"])],
            }
        )
        st.dataframe(metrics.style.format({"MAE": "{:.3f}°C", "RMSE": "{:.3f}°C"}), hide_index=True, width="stretch")
    with right:
        st.subheader("Notebook test benchmark")
        benchmark = pd.DataFrame(
            {
                "Model": ["LSTM · A · 72h"],
                "MAE": [1.229071],
                "RMSE": [1.614333],
                "Parameters": [21569],
            }
        )
        st.dataframe(benchmark.style.format({"MAE": "{:.3f}°C", "RMSE": "{:.3f}°C", "Parameters": "{:,.0f}"}), hide_index=True, width="stretch")

with methodology:
    st.markdown(
        """
        **What this dashboard means**

        Each plotted prediction is a +6-hour point forecast produced from a rolling 72-hour historical window. This view compares repeated point forecasts against their observed targets—it is not a native multi-step weather-model trajectory.

        **Current boundary**

        The models were trained on the Jena Climate dataset and require 19 engineered meteorological features. They should not be presented as live EGLC forecasts or Polymarket probabilities without retraining and validating on a compatible station/data pipeline.
        """
    )
    st.markdown('<div class="truth-note">The model was trained on the Jena Climate dataset. It should not be presented as a live EGLC forecast or Polymarket probability model without station-compatible retraining.</div>', unsafe_allow_html=True)
