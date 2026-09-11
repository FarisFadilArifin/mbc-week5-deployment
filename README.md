# TempSequence — LSTM v2 dashboard

A lightweight Streamlit dashboard for the LSTM v2 multi-horizon Jena Climate backtest. The controls, navigation, status badges, and metric cards use `streamlit-shadcn-ui`; Plotly remains in place for the interval and calibration charts.

The interface prioritizes +1h and +2h forecasts, while retaining +6h metrics for comparison with the v1 benchmark. It visualizes precomputed Kaggle exports; it does not load TensorFlow, call a weather API, or claim to be a live station or Kalshi forecast.

## Run locally

```powershell
cd D:\dev\mbc-week5-deployment\streamlit_v1
& .\.venv313\Scripts\python.exe -m streamlit run app.py
```

Then open `http://localhost:8501`.

The deployment pins Streamlit 1.63 and `streamlit-shadcn-ui` 1.4 because the shadcn Components V2 package requires Streamlit 1.60 or newer.

The app loads the bundled files from `artifacts/` automatically. Use the sidebar uploader to replace them with a compatible v2 export bundle.

## Bundle files

`predictions_v2.csv` is required. The other files are optional and enable their respective panels:

```text
predictions_v2.csv
threshold_predictions_v2.csv
experiment_results_v2.csv
calibration_summary_v2.csv
model_metadata_v2.json
```

Upload individual files or a ZIP archive containing these names. The app validates required columns, timestamp offsets, supported horizons, duplicate keys, numeric values, and probability bounds before displaying an upload.

## Prediction schema

```text
as_of_timestamp,target_timestamp,horizon_hours,actual_temperature_c,predicted_temperature_c,lower_10_c,upper_90_c
```

The target timestamp must equal the as-of timestamp plus the declared horizon in hours. Supported horizons are +1h, +2h, and +6h.

## Important boundary

The threshold view is a synthetic Kalshi-style demonstration produced from empirical validation residuals. It is not an official Kalshi probability, market price, trading signal, or live weather forecast.

The dashboard also derives mutually exclusive temperature brackets from the exported cumulative thresholds at runtime. For adjacent strikes, bracket mass is calculated as `P(T >= lower) - P(T >= upper)`, so the bracket chart can form a bell-shaped predictive distribution without rerunning the Kaggle notebook.

A future live-inference version would need a current 72-hour sequence with the exact 19-feature preprocessing pipeline used by the Kaggle notebook.
