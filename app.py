from __future__ import annotations

from io import BytesIO
from pathlib import Path
import html
import json
import os
import re
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="MBG Opinion Studio · IndoBERTweet v2",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
LABEL_ORDER = ["Negative", "Neutral", "Positive"]
LABEL_COLORS = {
    "Negative": "#ff6b7a",
    "Neutral": "#8394a8",
    "Positive": "#2fe3ae",
}
PROBABILITY_COLUMNS = {label: f"probability_{label}" for label in LABEL_ORDER}
MAX_COMMENT_LENGTH = 1000
V2_ARTIFACT_NAMES = {
    "test": ("predictions_test_v2.csv",),
    "oof": ("predictions_oof_v2.csv",),
    "folds": ("fold_metrics_v2.csv",),
    "report": ("classification_report_v2.csv",),
    "confusion": ("confusion_matrix_v2.csv",),
    "metadata": ("model_metadata_v2.json",),
    "labels": ("label_mapping_v2.json",),
}
PREDICTION_REQUIRED = {
    "id",
    "comment",
    "predicted_label",
    *PROBABILITY_COLUMNS.values(),
}


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
            --amber: #f1c66d;
        }
        .stApp { background: var(--bg); color: var(--text); }
        [data-testid="stHeader"] { background: rgba(7,10,14,.92); }
        [data-testid="stSidebar"] { background: #0b1015; border-right: 1px solid var(--line); }
        .block-container { max-width: 1380px; padding-top: 1.7rem; padding-bottom: 4rem; }
        .brand { font-size: 1.24rem; font-weight: 850; letter-spacing: -.04em; }
        .brand span { color: var(--mint); }
        .side-copy { color: var(--muted); font-size: .76rem; line-height: 1.45; }
        .eyebrow { color: var(--mint); font: 700 .66rem/1.2 ui-monospace, SFMono-Regular, monospace; letter-spacing: .17em; }
        .hero-title { font-size: clamp(2.05rem, 4vw, 3.65rem); line-height: 1.02; margin: .42rem 0 .55rem; letter-spacing: -.058em; }
        .hero-copy { color: var(--muted); max-width: 800px; font-size: .96rem; line-height: 1.55; }
        .source-line { color: var(--muted); font: 700 .66rem/1.2 ui-monospace, SFMono-Regular, monospace; letter-spacing: .08em; text-transform: uppercase; margin-top: .8rem; }
        .status-pill { display: inline-block; padding: .27rem .56rem; border-radius: 999px; font: 700 .63rem/1 ui-monospace, SFMono-Regular, monospace; letter-spacing: .05em; }
        .status-pill.green { color: #9dffdf; background: #0c2b24; border: 1px solid #1b755e; }
        .status-pill.amber { color: #f7dda1; background: #2b2110; border: 1px solid #866626; }
        .metric-card { background: linear-gradient(145deg, var(--panel-2), var(--panel)); border: 1px solid var(--line); border-radius: 12px; padding: .74rem .86rem; min-height: 101px; }
        .metric-label { color: var(--muted); font: 700 .62rem/1.2 ui-monospace, SFMono-Regular, monospace; letter-spacing: .1em; text-transform: uppercase; }
        .metric-value { font-size: 1.62rem; font-weight: 820; letter-spacing: -.04em; margin: .42rem 0 .12rem; }
        .metric-foot { color: var(--muted); font-size: .71rem; }
        .section-kicker { color: var(--cyan); font: 700 .64rem/1.2 ui-monospace, SFMono-Regular, monospace; letter-spacing: .14em; text-transform: uppercase; margin: 1.24rem 0 .42rem; }
        .input-panel { background: linear-gradient(145deg, #151e27, #10161d); border: 1px solid #355164; border-radius: 15px; padding: 1.15rem 1.2rem .95rem; }
        .input-title { font-size: 1.05rem; font-weight: 780; }
        .input-subtitle { color: var(--muted); font-size: .76rem; margin: .22rem 0 .8rem; }
        .result-panel { background: linear-gradient(145deg, #141c24, #10161d); border: 1px solid var(--line); border-radius: 14px; padding: 1.08rem 1.15rem; min-height: 220px; }
        .result-label { color: var(--muted); font: 700 .63rem/1.2 ui-monospace, SFMono-Regular, monospace; letter-spacing: .11em; text-transform: uppercase; }
        .result-sentiment { font-size: 1.85rem; font-weight: 850; letter-spacing: -.045em; margin: .34rem 0 .1rem; }
        .comment-panel { background: linear-gradient(145deg, #141c24, #10161d); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.05rem; min-height: 170px; }
        .comment-text { font-size: 1.02rem; line-height: 1.5; margin: .65rem 0 .9rem; color: #f3f7fa; }
        .prediction-chip { display: inline-block; border-radius: 999px; padding: .38rem .68rem; font-weight: 800; font-size: .76rem; }
        .panel-title { font-size: 1.01rem; font-weight: 760; margin: 0; }
        .panel-subtitle { color: var(--muted); font-size: .73rem; margin-top: .18rem; }
        .small-note { color: var(--muted); font-size: .73rem; line-height: 1.45; }
        .callout { background: #0d1a22; border-left: 3px solid var(--cyan); padding: .72rem .86rem; border-radius: 0 8px 8px 0; color: #bfd0dd; font-size: .76rem; line-height: 1.5; }
        .warning-callout { background: #2b2110; border-left: 3px solid var(--amber); padding: .72rem .86rem; border-radius: 0 8px 8px 0; color: #f4d99b; font-size: .76rem; line-height: 1.5; }
        .stTabs [data-baseweb="tab-list"] { gap: 1.05rem; border-bottom: 1px solid var(--line); }
        .stTabs [aria-selected="true"] { color: var(--mint) !important; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def secret_value(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.getenv(name, ""))


def clean_social_text(value: str) -> str:
    """Match the Kaggle notebook's URL/mention/whitespace normalization."""
    value = str(value).strip().lower()
    value = re.sub(r"https?://\S+|www\.\S+", "HTTPURL", value)
    value = re.sub(r"(?<!\w)@\w+", "@USER", value)
    try:
        import emoji

        value = emoji.demojize(value, delimiters=(" ", " "))
    except ImportError:
        pass
    return re.sub(r"\s+", " ", value).strip()


def canonical_name(name: str) -> str | None:
    base = Path(name).name.lower()
    if base.startswith("predictions_test_v2") and base.endswith(".csv"):
        return "test"
    if (base.startswith("predictions_oof_v2") or base.startswith("predictions_validation_v2")) and base.endswith(".csv"):
        return "oof"
    if base.startswith("fold_metrics_v2") and base.endswith(".csv"):
        return "folds"
    if base.startswith("classification_report_v2") and base.endswith(".csv"):
        return "report"
    if base.startswith("confusion_matrix_v2") and base.endswith(".csv"):
        return "confusion"
    if base.startswith("model_metadata_v2") and base.endswith(".json"):
        return "metadata"
    if base.startswith("label_mapping_v2") and base.endswith(".json"):
        return "labels"
    return None


def read_uploads(files) -> dict[str, bytes]:
    payload: dict[str, bytes] = {}
    for uploaded in files:
        if uploaded.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(BytesIO(uploaded.getvalue())) as archive:
                    for member in archive.infolist():
                        if not member.is_dir():
                            key = canonical_name(member.filename)
                            if key:
                                payload[key] = archive.read(member)
            except zipfile.BadZipFile as error:
                raise ValueError(f"{uploaded.name} bukan ZIP yang valid.") from error
        else:
            key = canonical_name(uploaded.name)
            if key:
                payload[key] = uploaded.getvalue()
    if "test" not in payload:
        raise ValueError("Upload harus berisi predictions_test_v2.csv.")
    return payload


def load_csv(payload: dict[str, bytes], key: str, names: tuple[str, ...], uploaded: bool) -> pd.DataFrame | None:
    if key in payload:
        return pd.read_csv(BytesIO(payload[key]))
    if uploaded:
        return None
    for name in names:
        path = ARTIFACT_DIR / name
        if path.is_file():
            return pd.read_csv(path)
    return None


def load_json(payload: dict[str, bytes], key: str, names: tuple[str, ...], uploaded: bool) -> dict:
    if key in payload:
        return json.loads(payload[key].decode("utf-8"))
    if uploaded:
        return {}
    for name in names:
        path = ARTIFACT_DIR / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def validate_predictions(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    missing = PREDICTION_REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"{context} tidak memiliki kolom: {', '.join(sorted(missing))}")
    frame = frame.copy()
    if frame.empty:
        raise ValueError(f"{context} kosong.")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"{context}: ID harus ada dan unik.")
    frame["comment"] = frame["comment"].fillna("").astype(str)
    frame["predicted_label"] = frame["predicted_label"].astype(str)
    unknown = set(frame["predicted_label"]) - set(LABEL_ORDER)
    if unknown:
        raise ValueError(f"{context}: label prediksi tidak dikenal: {sorted(unknown)}")
    for column in PROBABILITY_COLUMNS.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all() or not frame[column].between(0, 1).all():
            raise ValueError(f"{context}: {column} harus numerik dan berada di antara 0 dan 1.")
    sums = frame[list(PROBABILITY_COLUMNS.values())].sum(axis=1)
    if not np.allclose(sums.to_numpy(), 1.0, atol=0.03):
        raise ValueError(f"{context}: probabilitas kelas harus berjumlah sekitar 1.0.")
    return frame


def validate_oof(frame: pd.DataFrame) -> pd.DataFrame:
    if "label" not in frame.columns and "true_label" in frame.columns:
        frame = frame.rename(columns={"true_label": "label"})
    missing = PREDICTION_REQUIRED | {"label"}
    absent = missing.difference(frame.columns)
    if absent:
        raise ValueError(f"predictions_oof_v2.csv tidak memiliki kolom: {', '.join(sorted(absent))}")
    frame = validate_predictions(frame, "predictions_oof_v2.csv")
    frame["label"] = frame["label"].astype(str)
    unknown = set(frame["label"]) - set(LABEL_ORDER)
    if unknown:
        raise ValueError(f"predictions_oof_v2.csv: label aktual tidak dikenal: {sorted(unknown)}")
    return frame


def calculate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    actual = frame["label"].to_numpy()
    predicted = frame["predicted_label"].to_numpy()
    rows = []
    for label in LABEL_ORDER:
        tp = int(((actual == label) & (predicted == label)).sum())
        fp = int(((actual != label) & (predicted == label)).sum())
        fn = int(((actual == label) & (predicted != label)).sum())
        support = int((actual == label).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": label, "precision": precision, "recall": recall, "f1-score": f1, "support": support})
    per_class = pd.DataFrame(rows)
    accuracy = float((actual == predicted).mean())
    return pd.concat(
        [per_class, pd.DataFrame([
            {"label": "accuracy", "precision": accuracy, "recall": accuracy, "f1-score": accuracy, "support": len(frame)},
            {"label": "macro avg", "precision": per_class["precision"].mean(), "recall": per_class["recall"].mean(), "f1-score": per_class["f1-score"].mean(), "support": len(frame)},
        ])],
        ignore_index=True,
    )


def metric_value(metrics: pd.DataFrame | None, row: str, column: str) -> float | None:
    if metrics is None:
        return None
    values = metrics.loc[metrics["label"] == row, column]
    return float(values.iloc[0]) if not values.empty else None


def metric_card(label: str, value: str, foot: str, accent: str = "") -> None:
    style = f' style="color:{accent}"' if accent else ""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value"{style}>{value}</div><div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def distribution_chart(frame: pd.DataFrame) -> go.Figure:
    counts = frame["predicted_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    percentages = counts / max(int(counts.sum()), 1) * 100
    figure = go.Figure(go.Bar(
        x=LABEL_ORDER,
        y=counts.to_numpy(),
        text=[f"{value:.1f}%" for value in percentages],
        textposition="outside",
        marker_color=[LABEL_COLORS[label] for label in LABEL_ORDER],
        customdata=percentages.to_numpy(),
        hovertemplate="%{x}<br>%{y} komentar (%{customdata:.1f}%)<extra></extra>",
    ))
    figure.update_layout(
        height=320,
        margin={"l": 8, "r": 8, "t": 28, "b": 8},
        paper_bgcolor="#10161d",
        plot_bgcolor="#10161d",
        font={"color": "#b8c5d1", "family": "Inter, sans-serif"},
        xaxis={"showgrid": False},
        yaxis={"title": "Komentar", "gridcolor": "#27323e", "zeroline": False},
        showlegend=False,
    )
    return figure


def probability_chart(probabilities: dict[str, float]) -> go.Figure:
    labels = list(reversed(LABEL_ORDER))
    values = [probabilities[label] * 100 for label in labels]
    figure = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        text=[f"{value:.1f}%" for value in values],
        textposition="outside",
        marker_color=[LABEL_COLORS[label] for label in labels],
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    figure.update_layout(
        height=245,
        margin={"l": 5, "r": 38, "t": 8, "b": 8},
        paper_bgcolor="#141c24",
        plot_bgcolor="#141c24",
        font={"color": "#b8c5d1", "family": "Inter, sans-serif"},
        xaxis={"range": [0, 108], "ticksuffix": "%", "gridcolor": "#27323e"},
        yaxis={"showgrid": False},
        showlegend=False,
    )
    return figure


def confusion_chart(frame: pd.DataFrame) -> go.Figure:
    matrix = pd.crosstab(frame["label"], frame["predicted_label"]).reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)
    figure = go.Figure(go.Heatmap(
        z=matrix.to_numpy(),
        x=LABEL_ORDER,
        y=LABEL_ORDER,
        colorscale=[[0, "#111820"], [0.5, "#1d6a63"], [1, "#2fe3ae"]],
        text=matrix.to_numpy(),
        texttemplate="%{text}",
        hovertemplate="Aktual %{y}<br>Prediksi %{x}: %{z}<extra></extra>",
        showscale=False,
    ))
    figure.update_layout(
        height=320,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="#10161d",
        plot_bgcolor="#10161d",
        font={"color": "#b8c5d1", "family": "Inter, sans-serif"},
        xaxis={"title": "Prediksi"},
        yaxis={"title": "Aktual", "autorange": "reversed"},
    )
    return figure


def confidence_tier(confidence: float) -> tuple[str, str]:
    if confidence >= 0.80:
        return "TINGGI", LABEL_COLORS["Positive"]
    if confidence >= 0.60:
        return "PERLU REVIEW", "#f1c66d"
    return "RENDAH", LABEL_COLORS["Negative"]


def chip(label: str) -> str:
    text_color = "#06110e" if label == "Positive" else "#ffffff"
    return f'<span class="prediction-chip" style="background:{LABEL_COLORS[label]};color:{text_color};">{label}</span>'


@st.cache_resource(show_spinner=False)
def load_live_model(source: str, token: str):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Dependensi live inference belum tersedia. Install torch dan transformers dari requirements.txt."
        ) from error

    kwargs = {"token": token} if token else {}
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(source, **kwargs)
    model.to("cpu")
    model.eval()
    return tokenizer, model


def classify_text(text: str, source: str, token: str, max_length: int) -> dict:
    import torch

    tokenizer, model = load_live_model(source, token)
    cleaned = clean_social_text(text)
    inputs = tokenizer(
        cleaned,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    model_labels = getattr(model.config, "id2label", {})
    decoded = {
        int(key): str(value)
        for key, value in model_labels.items()
    }
    if not all(decoded.get(index) in LABEL_ORDER for index in range(len(probabilities))):
        decoded = {index: label for index, label in enumerate(LABEL_ORDER)}
    probability_map = {
        label: float(probabilities[index])
        for index, label in decoded.items()
        if label in LABEL_ORDER
    }
    probability_map = {label: probability_map.get(label, 0.0) for label in LABEL_ORDER}
    predicted = max(probability_map, key=probability_map.get)
    return {
        "original": text,
        "cleaned": cleaned,
        "label": predicted,
        "confidence": probability_map[predicted],
        "probabilities": probability_map,
    }


inject_styles()

with st.sidebar:
    st.markdown('<div class="brand">💬 MBG<span>Pulse</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-copy">Studio klasifikasi opini publik MBG berbasis IndoBERTweet v2.</div>', unsafe_allow_html=True)
    st.divider()
    uploads = st.file_uploader(
        "Upload artefak v2 (opsional)",
        type=["zip", "csv", "json"],
        accept_multiple_files=True,
        help="Gunakan ZIP hasil notebook v2 atau predictions_test_v2.csv.",
    )
    if uploads:
        try:
            uploaded_payload = read_uploads(uploads)
        except ValueError as error:
            st.error(str(error))
            st.stop()
        payload = uploaded_payload
        uploaded_mode = True
        source_label = "UPLOAD AKTIF"
    else:
        payload = {}
        uploaded_mode = False
        source_label = "BUNDLED V2"

    st.markdown('<div class="small-note">Komentar yang diketik tidak disimpan ke server. Hanya riwayat sesi browser yang ditampilkan.</div>', unsafe_allow_html=True)
    st.divider()
    if (ARTIFACT_DIR / "predictions_test_v2.csv").is_file() or "test" in payload:
        st.markdown(f'<span class="status-pill green">{source_label}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill amber">V2 ARTIFACT BELUM TERSEDIA</span>', unsafe_allow_html=True)
    local_model_dir = ARTIFACT_DIR / "indobertweet_model"
    hf_model_id = secret_value("HF_MODEL_ID")
    model_source = str(local_model_dir) if (local_model_dir / "config.json").is_file() else hf_model_id
    if model_source:
        st.markdown('<span class="status-pill green">MODEL SIAP SAAT DIMINTA</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill amber">MODEL BELUM DIKONFIGURASI</span>', unsafe_allow_html=True)
    if st.button("Hapus riwayat sesi", use_container_width=True):
        st.session_state.prediction_history = []


test_raw = load_csv(payload, "test", V2_ARTIFACT_NAMES["test"], uploaded_mode)
oof_raw = load_csv(payload, "oof", V2_ARTIFACT_NAMES["oof"], uploaded_mode)
folds_raw = load_csv(payload, "folds", V2_ARTIFACT_NAMES["folds"], uploaded_mode)
metadata = load_json(payload, "metadata", V2_ARTIFACT_NAMES["metadata"], uploaded_mode)
labels_metadata = load_json(payload, "labels", V2_ARTIFACT_NAMES["labels"], uploaded_mode)

predictions = None
oof = None
artifact_error = None
if test_raw is not None:
    try:
        predictions = validate_predictions(test_raw, "predictions_test_v2.csv")
    except ValueError as error:
        artifact_error = str(error)
if oof_raw is not None:
    try:
        oof = validate_oof(oof_raw)
    except ValueError as error:
        artifact_error = str(error)

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "comment_input" not in st.session_state:
    st.session_state.comment_input = ""

st.markdown('<div class="eyebrow">MBG PUBLIC OPINION LAB · INDOBERTWEET V2</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Klasifikasi opini, satu komentar sekali</h1>', unsafe_allow_html=True)
st.markdown('<div class="hero-copy">Tulis komentar tentang program Makan Bergizi Gratis dan lihat bagaimana model membaca kecenderungan sentimennya. Model live memakai best fold; analitik dataset memakai hasil ensemble lima fold.</div>', unsafe_allow_html=True)
st.markdown(f'<div class="source-line"><span class="status-pill green">{source_label}</span> &nbsp; · &nbsp; PRIVATE MODEL LOADER &nbsp; · &nbsp; NO PERSISTENT COMMENT LOG</div>', unsafe_allow_html=True)

metrics = calculate_metrics(oof) if oof is not None else None
if predictions is not None:
    distribution = predictions["predicted_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
else:
    distribution = pd.Series(0, index=LABEL_ORDER, dtype=int)

st.markdown('<div class="section-kicker">Ringkasan model</div>', unsafe_allow_html=True)
kpis = st.columns(4)
with kpis[0]:
    value = f"{metric_value(metrics, 'macro avg', 'f1-score'):.3f}" if metrics is not None else "—"
    metric_card("OOF macro F1", value, "5-fold validation", LABEL_COLORS["Positive"])
with kpis[1]:
    value = f"{metric_value(metrics, 'accuracy', 'f1-score'):.1%}" if metrics is not None else "—"
    metric_card("OOF accuracy", value, "bukan skor test tersembunyi", LABEL_COLORS["Positive"])
with kpis[2]:
    metric_card("Test comments", f"{len(predictions):,}" if predictions is not None else "—", "prediksi Kaggle v2", LABEL_COLORS["Neutral"])
with kpis[3]:
    metric_card("Most frequent", str(distribution.idxmax()) if int(distribution.sum()) else "—", f"{int(distribution.max()):,} komentar" if int(distribution.sum()) else "belum ada artefak", LABEL_COLORS[str(distribution.idxmax())] if int(distribution.sum()) else LABEL_COLORS["Neutral"])

tabs = st.tabs(["Klasifikasi", "Eksplorasi data", "Kualitas model", "Metodologi"])

with tabs[0]:
    st.markdown('<div class="section-kicker">Coba komentar Anda</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-panel"><div class="input-title">Apa pendapat Anda tentang MBG?</div><div class="input-subtitle">Gunakan bahasa Indonesia natural. Model mempertahankan emoji, mention, dan konteks sosial-media.</div></div>', unsafe_allow_html=True)
    example_cols = st.columns(3)
    examples = [
        "Program ini sangat membantu anak-anak di daerah.",
        "Semoga pelaksanaannya transparan dan tepat sasaran.",
        "Anggarannya besar, tapi kualitas makanannya mengecewakan.",
    ]
    for column, example in zip(example_cols, examples):
        with column:
            if st.button(example, key=f"example_{hash(example)}", use_container_width=True):
                st.session_state.comment_input = example

    comment = st.text_area(
        "Tulis komentar",
        key="comment_input",
        height=150,
        max_chars=MAX_COMMENT_LENGTH,
        placeholder="Contoh: semoga program MBG benar-benar sampai ke anak yang membutuhkan...",
        label_visibility="collapsed",
    )
    st.caption(f"{len(comment):,}/{MAX_COMMENT_LENGTH} karakter")
    classify_clicked = st.button("Klasifikasikan komentar", type="primary", use_container_width=True)

    if classify_clicked:
        if not comment.strip():
            st.warning("Tulis komentar terlebih dahulu.")
        elif not model_source:
            st.error("Model belum dikonfigurasi. Tambahkan HF_MODEL_ID dan HF_TOKEN di Streamlit secrets, atau letakkan folder artifacts/indobertweet_model secara lokal.")
        else:
            try:
                max_length = int(metadata.get("max_length", 160))
                with st.spinner("Memuat model dan menghitung prediksi..."):
                    result = classify_text(comment, model_source, secret_value("HF_TOKEN"), max_length)
                st.session_state.last_result = result
                st.session_state.prediction_history = [
                    result,
                    *st.session_state.prediction_history,
                ][:20]
            except Exception as error:
                st.error(f"Prediksi gagal: {error}")

    if "last_result" in st.session_state:
        result = st.session_state.last_result
        tier, tier_color = confidence_tier(result["confidence"])
        st.markdown('<div class="section-kicker">Hasil prediksi</div>', unsafe_allow_html=True)
        result_left, result_right = st.columns([1.0, 1.12], gap="large")
        with result_left:
            st.markdown(
                f'<div class="result-panel"><div class="result-label">Sentimen terpilih</div><div class="result-sentiment" style="color:{LABEL_COLORS[result["label"]]};">{result["label"]}</div><div class="small-note">Confidence {result["confidence"]:.1%} · tingkat {tier.lower()}</div><div style="margin-top:1rem;display:inline-block;padding:.27rem .55rem;border:1px solid {tier_color};border-radius:999px;color:{tier_color};font:700 .64rem ui-monospace,monospace;">KEYAKINAN {tier}</div><div class="small-note" style="margin-top:1rem;">Teks yang dinormalisasi model:<br>{html.escape(result["cleaned"])}</div></div>',
                unsafe_allow_html=True,
            )
        with result_right:
            st.markdown('<div class="panel-title">Profil probabilitas</div><div class="panel-subtitle">Distribusi softmax untuk tiga kelas sentimen</div>', unsafe_allow_html=True)
            st.plotly_chart(probability_chart(result["probabilities"]), width="stretch", config={"displayModeBar": False})

    st.markdown('<div class="section-kicker">Riwayat sesi</div>', unsafe_allow_html=True)
    if st.session_state.prediction_history:
        history = pd.DataFrame([
            {"Komentar": item["original"], "Prediksi": item["label"], "Confidence": item["confidence"]}
            for item in st.session_state.prediction_history
        ])
        st.dataframe(
            history,
            hide_index=True,
            width="stretch",
            height=260,
            column_config={
                "Komentar": st.column_config.TextColumn(width="large"),
                "Confidence": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
            },
        )
    else:
        st.markdown('<div class="small-note">Belum ada prediksi pada sesi ini.</div>', unsafe_allow_html=True)

with tabs[1]:
    if predictions is None:
        st.markdown('<div class="warning-callout"><b>V2 artifacts belum tersedia.</b><br>Jalankan notebook five-fold di Kaggle, lalu upload predictions_test_v2.csv atau ZIP hasil ekspor untuk mengaktifkan eksplorasi.</div>', unsafe_allow_html=True)
    else:
        explore_left, explore_right = st.columns([1.35, 1], gap="large")
        with explore_left:
            st.markdown('<div class="panel-title">Distribusi prediksi test</div><div class="panel-subtitle">Test Kaggle tidak memiliki label publik; grafik ini adalah output model, bukan hasil survei.</div>', unsafe_allow_html=True)
            st.plotly_chart(distribution_chart(predictions), width="stretch", config={"displayModeBar": False})
        with explore_right:
            st.markdown('<div class="panel-title">Filter komentar</div><div class="panel-subtitle">Pilih label dan cari kata di komentar asli</div>', unsafe_allow_html=True)
            explore_labels = st.multiselect("Sentimen", LABEL_ORDER, default=LABEL_ORDER, key="explore_labels")
            explore_query = st.text_input("Cari", placeholder="contoh: sekolah, gizi, anggaran", key="explore_query")
            explore_count = st.slider("Jumlah baris", 10, 100, 25, step=5, key="explore_count")
        explore = predictions[predictions["predicted_label"].isin(explore_labels)].copy()
        if explore_query:
            explore = explore[explore["comment"].str.contains(explore_query, case=False, na=False, regex=False)]
        if st.button("↻ Pilih komentar acak", key="random_test_comment") and not explore.empty:
            random_row = explore.sample(1).iloc[0]
            st.session_state.comment_input = str(random_row["comment"])
            st.session_state.last_result = {
                "original": str(random_row["comment"]),
                "cleaned": clean_social_text(str(random_row["comment"])),
                "label": str(random_row["predicted_label"]),
                "confidence": float(random_row[PROBABILITY_COLUMNS[str(random_row["predicted_label"])] ]),
                "probabilities": {label: float(random_row[PROBABILITY_COLUMNS[label]]) for label in LABEL_ORDER},
            }
            st.rerun()
        table = explore.head(explore_count).copy()
        table["confidence"] = [float(row[PROBABILITY_COLUMNS[str(row["predicted_label"])]]) for _, row in table.iterrows()]
        table = table[["id", "comment", "predicted_label", "confidence"]]
        table.columns = ["ID", "Komentar", "Prediksi", "Confidence"]
        st.dataframe(table, hide_index=True, width="stretch", height=420, column_config={"Komentar": st.column_config.TextColumn(width="large"), "Confidence": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1)})
        if explore.empty:
            st.info("Tidak ada baris yang cocok dengan filter.")

with tabs[2]:
    if oof is None:
        st.markdown('<div class="warning-callout"><b>OOF validation belum tersedia.</b><br>Upload predictions_oof_v2.csv untuk menampilkan metrik dan confusion matrix.</div>', unsafe_allow_html=True)
    else:
        quality_left, quality_right = st.columns([1.1, 1], gap="large")
        with quality_left:
            st.markdown('<div class="panel-title">Confusion matrix OOF</div><div class="panel-subtitle">Setiap baris training dievaluasi oleh fold yang tidak melihatnya saat training</div>', unsafe_allow_html=True)
            st.plotly_chart(confusion_chart(oof), width="stretch", config={"displayModeBar": False})
        with quality_right:
            st.markdown('<div class="panel-title">Performa per kelas</div><div class="panel-subtitle">Macro F1 adalah metrik utama model</div>', unsafe_allow_html=True)
            display_metrics = metrics[metrics["label"].isin(LABEL_ORDER + ["macro avg", "accuracy"])].rename(columns={"label": "Kelas", "f1-score": "F1"})
            st.dataframe(display_metrics[["Kelas", "precision", "recall", "F1", "support"]].style.format({"precision": "{:.1%}", "recall": "{:.1%}", "F1": "{:.1%}"}), hide_index=True, width="stretch", height=280)
        if folds_raw is not None and {"fold", "macro_f1"}.issubset(folds_raw.columns):
            fold_plot = folds_raw.copy()
            fold_plot["fold"] = fold_plot["fold"].astype(str)
            fold_plot["macro_f1"] = pd.to_numeric(fold_plot["macro_f1"], errors="coerce")
            figure = go.Figure(go.Bar(x=fold_plot["fold"], y=fold_plot["macro_f1"], marker_color="#2fe3ae", text=fold_plot["macro_f1"].map(lambda value: f"{value:.3f}"), textposition="outside"))
            figure.update_layout(height=260, margin={"l": 8, "r": 8, "t": 25, "b": 8}, paper_bgcolor="#10161d", plot_bgcolor="#10161d", font={"color": "#b8c5d1"}, yaxis={"range": [0, 1], "gridcolor": "#27323e", "title": "Macro F1"}, xaxis={"title": "Fold"}, showlegend=False)
            st.markdown('<div class="section-kicker">Skor setiap fold</div>', unsafe_allow_html=True)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

with tabs[3]:
    metadata_display = {
        "Versi": metadata.get("model_version", "indobertweet_mbg_v2"),
        "Base model": metadata.get("base_model", "indolem/indobertweet-base-uncased"),
        "Validasi": f"{metadata.get('n_splits', 5)}-fold stratified",
        "Model live": "Best fold",
        "Test ensemble": metadata.get("ensemble_prediction_method", "mean softmax probabilities"),
        "Max length": metadata.get("max_length", 160),
        "Weight decay": metadata.get("weight_decay", "—"),
        "Label smoothing": metadata.get("label_smoothing", "—"),
        "Dropout classifier": metadata.get("classifier_dropout", "—"),
    }
    st.markdown('<div class="panel-title">Konfigurasi model</div><div class="panel-subtitle">Metadata berasal dari ekspor notebook v2</div>', unsafe_allow_html=True)
    metadata_table = pd.DataFrame(list(metadata_display.items()), columns=["Field", "Value"])
    metadata_table["Value"] = metadata_table["Value"].astype(str)
    st.dataframe(metadata_table, hide_index=True, width="stretch", height=330)
    st.markdown('<div class="section-kicker">Batas interpretasi</div>', unsafe_allow_html=True)
    st.markdown('<div class="callout">Prediksi live adalah keluaran model, bukan keputusan moderasi atau kebenaran objektif tentang isi komentar. Distribusi test Kaggle tidak memiliki label publik sehingga tidak boleh disebut sebagai polling opini. Komentar pengguna hanya berada di memori sesi Streamlit.</div>', unsafe_allow_html=True)
    if artifact_error:
        st.markdown(f'<div class="warning-callout">Validasi artefak: {html.escape(artifact_error)}</div>', unsafe_allow_html=True)
