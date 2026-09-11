from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="MBG Opinion Pulse",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
LABEL_ORDER = ["Negative", "Neutral", "Positive"]
LABEL_COLORS = {
    "Negative": "#ff6b7a",
    "Neutral": "#7f91a8",
    "Positive": "#2fe3ae",
}
PROBABILITY_COLUMNS = {
    label: f"probability_{label}" for label in LABEL_ORDER
}
PREDICTION_REQUIRED = {
    "id",
    "comment",
    "predicted_label",
    *PROBABILITY_COLUMNS.values(),
}
VALIDATION_REQUIRED = PREDICTION_REQUIRED | {"label"}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #070a0e;
            --panel: #10161d;
            --panel-2: #141c24;
            --line: #27323e;
            --text: #edf3f8;
            --muted: #8b9aaa;
            --mint: #2fe3ae;
            --cyan: #75d8ff;
        }
        .stApp { background: var(--bg); color: var(--text); }
        [data-testid="stHeader"] { background: rgba(7,10,14,.88); }
        [data-testid="stSidebar"] { background: #0b1015; border-right: 1px solid var(--line); }
        .block-container { max-width: 1380px; padding-top: 2rem; padding-bottom: 4rem; }
        .brand { font-size: 1.25rem; font-weight: 800; letter-spacing: -.04em; }
        .brand span { color: var(--mint); }
        .side-copy { color: var(--muted); font-size: .78rem; line-height: 1.45; }
        .eyebrow { color: var(--mint); font: 700 .68rem/1.2 ui-monospace, SFMono-Regular, monospace; letter-spacing: .16em; }
        .hero-title { font-size: clamp(2.1rem, 4vw, 3.8rem); line-height: 1.02; margin: .45rem 0 .55rem; letter-spacing: -.055em; }
        .hero-copy { color: var(--muted); max-width: 780px; font-size: .98rem; line-height: 1.55; }
        .source-line { color: var(--muted); font: 700 .68rem/1.2 ui-monospace, SFMono-Regular, monospace; letter-spacing: .08em; text-transform: uppercase; margin-top: .8rem; }
        .status-pill { display: inline-block; padding: .28rem .58rem; border-radius: 999px; font: 700 .66rem/1 ui-monospace, SFMono-Regular, monospace; letter-spacing: .05em; }
        .status-pill.green { color: #9dffdf; background: #0c2b24; border: 1px solid #1b755e; }
        .metric-card { background: linear-gradient(145deg, var(--panel-2), var(--panel)); border: 1px solid var(--line); border-radius: 12px; padding: .8rem .9rem; min-height: 106px; }
        .metric-label { color: var(--muted); font: 700 .64rem/1.2 ui-monospace, SFMono-Regular, monospace; letter-spacing: .11em; text-transform: uppercase; }
        .metric-value { font-size: 1.7rem; font-weight: 800; letter-spacing: -.04em; margin: .44rem 0 .14rem; }
        .metric-foot { color: var(--muted); font-size: .73rem; }
        .section-kicker { color: var(--cyan); font: 700 .66rem/1.2 ui-monospace, SFMono-Regular, monospace; letter-spacing: .14em; text-transform: uppercase; margin: 1.25rem 0 .38rem; }
        .comment-panel { background: linear-gradient(145deg, #141c24, #10161d); border: 1px solid var(--line); border-radius: 14px; padding: 1.15rem 1.2rem; min-height: 206px; }
        .comment-label { color: var(--muted); font: 700 .65rem/1.2 ui-monospace, SFMono-Regular, monospace; letter-spacing: .12em; text-transform: uppercase; }
        .comment-text { font-size: 1.12rem; line-height: 1.52; margin: .75rem 0 1rem; color: #f3f7fa; }
        .prediction-chip { display: inline-block; border-radius: 999px; padding: .42rem .75rem; font-weight: 800; font-size: .8rem; }
        .small-note { color: var(--muted); font-size: .76rem; line-height: 1.45; }
        .panel-title { font-size: 1.03rem; font-weight: 760; margin: 0; }
        .panel-subtitle { color: var(--muted); font-size: .75rem; margin-top: .2rem; }
        .callout { background: #0d1a22; border-left: 3px solid var(--cyan); padding: .75rem .9rem; border-radius: 0 8px 8px 0; color: #bfd0dd; font-size: .78rem; line-height: 1.5; }
        .warning-callout { background: #2b2110; border-left: 3px solid #eebc4e; padding: .75rem .9rem; border-radius: 0 8px 8px 0; color: #f4d99b; font-size: .78rem; line-height: 1.5; }
        .stTabs [data-baseweb="tab-list"] { gap: 1.1rem; border-bottom: 1px solid var(--line); }
        .stTabs [aria-selected="true"] { color: var(--mint) !important; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def canonical_name(name: str) -> str | None:
    lowered = Path(name).name.lower()
    if lowered.startswith("predictions_test") and lowered.endswith(".csv"):
        return "test"
    if lowered.startswith("predictions_validation") and lowered.endswith(".csv"):
        return "validation"
    if lowered.startswith("classification_report") and lowered.endswith(".csv"):
        return "report"
    if lowered.startswith("confusion_matrix") and lowered.endswith(".csv"):
        return "confusion"
    if lowered.startswith("label_mapping") and lowered.endswith(".json"):
        return "labels"
    if lowered.startswith("model_metadata") and lowered.endswith(".json"):
        return "metadata"
    if lowered == "submission.csv" or lowered.startswith("submission ("):
        return "submission"
    return None


def read_uploads(uploaded_files) -> dict[str, bytes]:
    payload: dict[str, bytes] = {}
    for uploaded in uploaded_files:
        filename = uploaded.name.lower()
        if filename.endswith(".zip"):
            try:
                with zipfile.ZipFile(BytesIO(uploaded.getvalue())) as archive:
                    for member in archive.infolist():
                        if member.is_dir():
                            continue
                        key = canonical_name(member.filename)
                        if key:
                            payload[key] = archive.read(member)
            except zipfile.BadZipFile as error:
                raise ValueError(f"{uploaded.name} is not a valid ZIP file.") from error
        else:
            key = canonical_name(uploaded.name)
            if key:
                payload[key] = uploaded.getvalue()
    if not payload:
        raise ValueError(
            "No recognized export files were found. Upload the Kaggle ZIP or predictions_test.csv."
        )
    return payload


def read_csv_source(payload: dict[str, bytes], key: str, path: Path | None) -> pd.DataFrame | None:
    if key in payload:
        return pd.read_csv(BytesIO(payload[key]))
    if path is not None and path.is_file():
        return pd.read_csv(path)
    return None


def read_json_source(payload: dict[str, bytes], key: str, path: Path | None) -> dict:
    if key in payload:
        return json.loads(payload[key].decode("utf-8"))
    if path is not None and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def validate_probabilities(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    for column in PROBABILITY_COLUMNS.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{context}: {column} contains missing or non-numeric values.")
        if not frame[column].between(0, 1).all():
            raise ValueError(f"{context}: {column} must be between 0 and 1.")
    probability_sum = frame[list(PROBABILITY_COLUMNS.values())].sum(axis=1)
    if not np.allclose(probability_sum.to_numpy(), 1.0, atol=0.03):
        raise ValueError(f"{context}: class probabilities should sum to approximately 1.0.")
    return frame


def validate_predictions(frame: pd.DataFrame, context: str = "predictions_test.csv") -> pd.DataFrame:
    missing = PREDICTION_REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"{context} is missing columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    if frame.empty:
        raise ValueError(f"{context} contains no rows.")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"{context}: IDs must be present and unique.")
    frame["comment"] = frame["comment"].fillna("").astype(str)
    frame["predicted_label"] = frame["predicted_label"].astype(str)
    unknown = set(frame["predicted_label"]) - set(LABEL_ORDER)
    if unknown:
        raise ValueError(f"{context}: unknown predicted labels: {sorted(unknown)}")
    return validate_probabilities(frame, context)


def validate_validation(frame: pd.DataFrame) -> pd.DataFrame:
    missing = VALIDATION_REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(
            "predictions_validation.csv is missing columns: "
            + ", ".join(sorted(missing))
        )
    frame = validate_predictions(frame, "predictions_validation.csv")
    frame["label"] = frame["label"].astype(str)
    unknown = set(frame["label"]) - set(LABEL_ORDER)
    if unknown:
        raise ValueError(f"predictions_validation.csv: unknown true labels: {sorted(unknown)}")
    return frame


def calculate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    y_true = frame["label"].to_numpy()
    y_pred = frame["predicted_label"].to_numpy()
    rows = []
    for label in LABEL_ORDER:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        support = int((y_true == label).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": label, "precision": precision, "recall": recall, "f1-score": f1, "support": support})
    result = pd.DataFrame(rows)
    accuracy = float((y_true == y_pred).mean())
    return pd.concat(
        [
            result,
            pd.DataFrame([
                {"label": "accuracy", "precision": accuracy, "recall": accuracy, "f1-score": accuracy, "support": len(frame)},
                {"label": "macro avg", "precision": result["precision"].mean(), "recall": result["recall"].mean(), "f1-score": result["f1-score"].mean(), "support": len(frame)},
            ]),
        ],
        ignore_index=True,
    )


def metric_value(metrics: pd.DataFrame, row: str, column: str) -> float | None:
    subset = metrics.loc[metrics["label"].astype(str) == row, column]
    return float(subset.iloc[0]) if not subset.empty else None


def metric_card(label: str, value: str, foot: str, accent: str = "") -> None:
    style = f" style=\"color:{accent}\"" if accent else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value"{style}>{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def distribution_chart(frame: pd.DataFrame) -> go.Figure:
    counts = frame["predicted_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    total = max(int(counts.sum()), 1)
    percentages = counts / total * 100
    figure = go.Figure(
        go.Bar(
            x=LABEL_ORDER,
            y=counts.values,
            text=[f"{value:.1f}%" for value in percentages.values],
            textposition="outside",
            marker_color=[LABEL_COLORS[label] for label in LABEL_ORDER],
            hovertemplate="%{x}<br>%{y} comments (%{customdata:.1f}%)<extra></extra>",
            customdata=percentages.values,
        )
    )
    figure.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        paper_bgcolor="#10161d",
        plot_bgcolor="#10161d",
        font={"color": "#b8c5d1", "family": "Inter, sans-serif"},
        xaxis={"showgrid": False},
        yaxis={"title": "Comments", "gridcolor": "#27323e", "zeroline": False},
        showlegend=False,
    )
    return figure


def probability_chart(row: pd.Series) -> go.Figure:
    labels = list(reversed(LABEL_ORDER))
    values = [float(row[PROBABILITY_COLUMNS[label]]) * 100 for label in labels]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            text=[f"{value:.1f}%" for value in values],
            textposition="outside",
            marker_color=[LABEL_COLORS[label] for label in labels],
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
        )
    )
    figure.update_layout(
        height=250,
        margin={"l": 5, "r": 38, "t": 8, "b": 8},
        paper_bgcolor="#141c24",
        plot_bgcolor="#141c24",
        font={"color": "#b8c5d1", "family": "Inter, sans-serif"},
        xaxis={"range": [0, 108], "showgrid": True, "gridcolor": "#27323e", "ticksuffix": "%"},
        yaxis={"showgrid": False},
        showlegend=False,
    )
    return figure


def confusion_chart(frame: pd.DataFrame) -> go.Figure:
    matrix = pd.crosstab(frame["label"], frame["predicted_label"])
    matrix = matrix.reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=LABEL_ORDER,
            y=LABEL_ORDER,
            colorscale=[[0, "#111820"], [0.5, "#1d6a63"], [1, "#2fe3ae"]],
            text=matrix.to_numpy(),
            texttemplate="%{text}",
            hovertemplate="Actual %{y}<br>Predicted %{x}: %{z}<extra></extra>",
            showscale=False,
        )
    )
    figure.update_layout(
        height=330,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="#10161d",
        plot_bgcolor="#10161d",
        font={"color": "#b8c5d1", "family": "Inter, sans-serif"},
        xaxis={"title": "Predicted", "side": "bottom"},
        yaxis={"title": "Actual", "autorange": "reversed"},
    )
    return figure


def chip(label: str) -> str:
    color = LABEL_COLORS[label]
    text_color = "#06110e" if label == "Positive" else "#ffffff"
    return f'<span class="prediction-chip" style="background:{color};color:{text_color};">{label}</span>'


@st.cache_data(show_spinner=False)
def load_default_test() -> pd.DataFrame:
    return validate_predictions(pd.read_csv(ARTIFACT_DIR / "predictions_test.csv"))


inject_styles()


with st.sidebar:
    st.markdown('<div class="brand">💬 MBG<span>Pulse</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-copy">IndoBERTweet public-opinion explorer for the Makan Bergizi Gratis discussion.</div>', unsafe_allow_html=True)
    st.divider()
    uploads = st.file_uploader(
        "Upload Kaggle export",
        type=["zip", "csv", "json"],
        accept_multiple_files=True,
        help="Upload indobertweet_mbg_v1.zip or predictions_test.csv. Uploaded data replaces the bundled export.",
    )
    st.markdown('<div class="small-note">The ZIP may contain the model, but this lightweight dashboard reads the exported predictions and metrics only.</div>', unsafe_allow_html=True)

    if uploads:
        try:
            uploaded_payload = read_uploads(uploads)
        except ValueError as error:
            st.error(str(error))
            st.stop()
        if "test" not in uploaded_payload:
            st.error("The upload must include predictions_test.csv.")
            st.stop()
        source_payload = uploaded_payload
        source_label = "UPLOADED KAGGLE EXPORT"
    else:
        source_payload = {}
        source_label = "BUNDLED KAGGLE EXPORT"

    st.divider()
    st.markdown("**Explore the test predictions**")
    search_text = st.text_input("Search comments", placeholder="e.g. MBG, sekolah, gratis")
    selected_labels = st.multiselect("Sentiment", LABEL_ORDER, default=LABEL_ORDER)
    max_rows = st.slider("Rows in table", 10, 100, 25, step=5)


try:
    if uploads:
        predictions = validate_predictions(
            read_csv_source(source_payload, "test", None),
            "predictions_test.csv",
        )
    else:
        predictions = load_default_test()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load predictions: {error}")
    st.stop()

validation_raw = read_csv_source(
    source_payload,
    "validation",
    None if uploads else ARTIFACT_DIR / "predictions_validation.csv",
)
validation = None
if validation_raw is not None:
    try:
        validation = validate_validation(validation_raw)
    except ValueError as error:
        if uploads:
            st.error(str(error))
            st.stop()
        st.warning(str(error))

metadata = read_json_source(
    source_payload,
    "metadata",
    None if uploads else ARTIFACT_DIR / "model_metadata.json",
)

filtered = predictions.copy()
if selected_labels:
    filtered = filtered[filtered["predicted_label"].isin(selected_labels)]
else:
    filtered = filtered.iloc[0:0]
if search_text:
    filtered = filtered[
        filtered["comment"].str.contains(search_text, case=False, na=False, regex=False)
    ]

if "selected_id" not in st.session_state or st.session_state.selected_id not in set(predictions["id"]):
    st.session_state.selected_id = predictions.iloc[0]["id"]
if st.sidebar.button("↻ Random comment", use_container_width=True):
    pool = filtered if not filtered.empty else predictions
    st.session_state.selected_id = pool.sample(1).iloc[0]["id"]

selected_matches = predictions[predictions["id"] == st.session_state.selected_id]
selected = selected_matches.iloc[0] if not selected_matches.empty else predictions.iloc[0]
selected_label = str(selected["predicted_label"])
selected_confidence = float(selected[PROBABILITY_COLUMNS[selected_label]])

st.markdown('<div class="eyebrow">MBG PUBLIC OPINION LAB · INDOBERTWEET</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">What people are saying about MBG</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Explore sentiment predictions from the Kaggle test export, inspect model confidence, and review validation behavior in one compact research console.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="source-line"><span class="status-pill green">{source_label}</span> &nbsp; {len(predictions):,} TEST COMMENTS &nbsp; · &nbsp; 3 SENTIMENT CLASSES</div>',
    unsafe_allow_html=True,
)

metrics = calculate_metrics(validation) if validation is not None else None
distribution = predictions["predicted_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
dominant_label = str(distribution.idxmax())

st.markdown('<div class="section-kicker">Snapshot</div>', unsafe_allow_html=True)
kpi = st.columns(4)
with kpi[0]:
    value = f"{metric_value(metrics, 'macro avg', 'f1-score'):.3f}" if metrics is not None else "—"
    metric_card("Validation macro F1", value, "held-out labeled split", LABEL_COLORS["Positive"])
with kpi[1]:
    value = f"{metric_value(metrics, 'accuracy', 'f1-score'):.1%}" if metrics is not None else "—"
    metric_card("Validation accuracy", value, "not a live score", LABEL_COLORS["Positive"])
with kpi[2]:
    metric_card("Largest predicted class", dominant_label, f"{int(distribution.max()):,} of {len(predictions):,} comments", LABEL_COLORS[dominant_label])
with kpi[3]:
    metric_card("Selected confidence", f"{selected_confidence:.1%}", "probability of current label", LABEL_COLORS[selected_label])

tabs = st.tabs(["Opinion overview", "Model quality", "Data & export"])

with tabs[0]:
    st.markdown('<div class="section-kicker">Selected comment</div>', unsafe_allow_html=True)
    left, right = st.columns([1.03, 1.0], gap="large")
    with left:
        safe_comment = str(selected["comment"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f"""
            <div class="comment-panel">
                <div class="comment-label">Prediction record · ID {selected['id']}</div>
                <div class="comment-text">{safe_comment}</div>
                {chip(selected_label)}
                <span class="small-note"> &nbsp; {selected_confidence:.1%} confidence</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="panel-title">Class probability profile</div><div class="panel-subtitle">Softmax output for the selected comment</div>', unsafe_allow_html=True)
        st.plotly_chart(probability_chart(selected), width="stretch", config={"displayModeBar": False})

    st.markdown('<div class="section-kicker">Predicted sentiment mix</div>', unsafe_allow_html=True)
    chart_col, summary_col = st.columns([1.65, 1], gap="large")
    with chart_col:
        st.plotly_chart(distribution_chart(predictions), width="stretch", config={"displayModeBar": False})
    with summary_col:
        st.markdown('<div class="panel-title">Reading the mix</div><div class="panel-subtitle">Predicted labels across the unlabeled Kaggle test export</div>', unsafe_allow_html=True)
        for label in LABEL_ORDER:
            count = int(distribution[label])
            percent = count / max(len(predictions), 1)
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:.62rem 0;border-bottom:1px solid #27323e;"><span style="color:{LABEL_COLORS[label]};font-weight:750;">{label}</span><span>{count:,} <span class="small-note">({percent:.1%})</span></span></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="small-note" style="margin-top:1rem;">Because the Kaggle test labels are hidden, this is a prediction distribution—not a measured opinion poll.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Explore records</div>', unsafe_allow_html=True)
    table = filtered.head(max_rows).copy()
    table["confidence"] = [float(row[PROBABILITY_COLUMNS[str(row["predicted_label"])]]) for _, row in table.iterrows()]
    table = table[["id", "comment", "predicted_label", "confidence"]]
    table.columns = ["ID", "Comment", "Prediction", "Confidence"]
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=390,
        column_config={
            "Comment": st.column_config.TextColumn(width="large"),
            "Confidence": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
        },
    )
    if filtered.empty:
        st.info("No rows match the current sentiment/search filters.")

with tabs[1]:
    if validation is None or metrics is None:
        st.markdown('<div class="warning-callout">Validation artifacts are not available. Upload predictions_validation.csv to show quality diagnostics.</div>', unsafe_allow_html=True)
    else:
        quality_left, quality_right = st.columns([1.1, 1], gap="large")
        with quality_left:
            st.markdown('<div class="panel-title">Validation confusion matrix</div><div class="panel-subtitle">Rows are actual labels; columns are model predictions</div>', unsafe_allow_html=True)
            st.plotly_chart(confusion_chart(validation), width="stretch", config={"displayModeBar": False})
        with quality_right:
            st.markdown('<div class="panel-title">Per-class performance</div><div class="panel-subtitle">Metrics calculated from the labeled validation split</div>', unsafe_allow_html=True)
            display_metrics = metrics.copy()
            display_metrics = display_metrics[display_metrics["label"].isin(LABEL_ORDER + ["macro avg", "accuracy"])]
            display_metrics = display_metrics.rename(columns={"label": "Class", "f1-score": "F1"})
            st.dataframe(
                display_metrics[["Class", "precision", "recall", "F1", "support"]].style.format({"precision": "{:.1%}", "recall": "{:.1%}", "F1": "{:.1%}"}),
                hide_index=True,
                width="stretch",
                height=280,
            )
            st.markdown('<div class="callout">Macro F1 gives each sentiment equal weight. It is the primary local metric for this classification task.</div>', unsafe_allow_html=True)

with tabs[2]:
    export_left, export_right = st.columns([1, 1], gap="large")
    with export_left:
        st.markdown('<div class="panel-title">Download filtered predictions</div><div class="panel-subtitle">The export reflects the current search and sentiment filters</div>', unsafe_allow_html=True)
        st.download_button(
            "Download predictions CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="mbg_predictions_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown('<div class="small-note" style="margin-top:.75rem;">For the original Kaggle submission, use the bundled submission.csv separately.</div>', unsafe_allow_html=True)
    with export_right:
        st.markdown('<div class="panel-title">Model metadata</div><div class="panel-subtitle">Configuration recorded by the Kaggle export</div>', unsafe_allow_html=True)
        metadata_rows = {
            "Model": metadata.get("base_model", "IndoBERTweet"),
            "Version": metadata.get("model_version", "indobertweet_mbg_v1"),
            "Text column": metadata.get("text_column", "comment"),
            "Validation rows": metadata.get("validation_rows", "—"),
            "Test rows": metadata.get("competition_test_rows", len(predictions)),
        }
        metadata_table = pd.DataFrame(
            list(metadata_rows.items()),
            columns=["Field", "Value"],
        )
        metadata_table["Value"] = metadata_table["Value"].astype(str)
        st.dataframe(metadata_table, hide_index=True, width="stretch", height=230)

    st.markdown('<div class="section-kicker">Interpretation boundary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="callout">This is an artifact-backed Kaggle research dashboard. The competition test labels are hidden, so test-set predictions cannot be scored here. It does not claim to be a live public-opinion monitor or a real-time classifier.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="small-note" style="margin-top:1rem;">To add live inference later, deploy the full IndoBERTweet model separately or provide a compatible inference service. The 442 MB model weights are intentionally not loaded during dashboard startup.</div>', unsafe_allow_html=True)
