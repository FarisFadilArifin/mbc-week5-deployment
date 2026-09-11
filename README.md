# TempSequence — LSTM dashboard

A HuskyWeather-inspired Streamlit dashboard for the Week 3 Jena Climate LSTM experiment.

The dashboard visualizes repeated **+6-hour LSTM point forecasts** against observed hourly mean temperature. It is a backtest/evaluation dashboard, not a live EGLC forecast or a Polymarket probability model.

This repository keeps each assignment deliverable on its own branch:

| Deliverable | Branch | Streamlit app |
| --- | --- | --- |
| LSTM temperature v1 | `lstm-temperature-v1` | [lstm-temperature-v1.streamlit.app](https://lstm-temperature-v1.streamlit.app/) |
| LSTM temperature v2 | `lstm-temperature-v2` | [lstm-temperature-v2.streamlit.app](https://lstm-temperature-v2.streamlit.app/) |
| IndoBERTweet MBG v1 | `indobertweet-mbg-v1` | [indobertweet-mbg-v1.streamlit.app](https://indobertweet-mbg-v1.streamlit.app/) |
| IndoBERTweet MBG v2 | `indobertweet-mbg-v2` | [indobertweet-mbg-v2.streamlit.app](https://indobertweet-mbg-v2.streamlit.app/) |

The `main` branch is the baseline LSTM v1 snapshot. Use the named branches for the corresponding deployed version.

## Run locally

```powershell
cd D:\dev\mbc-week5-deployment\streamlit_v1
& .\.venv313\Scripts\python.exe -m streamlit run app.py
```

Then open `http://localhost:8501`.

If `artifacts/predictions_v1.csv` exists, the app loads it automatically. You can also upload another CSV from the sidebar.

## Prediction CSV format

```text
timestamp,actual_temperature,lstm_prediction
2016-01-04 06:00:00,1.20,1.03
```

Extra columns are ignored. The required columns are `timestamp`, `actual_temperature`, and `lstm_prediction`.

## Kaggle export cell

Run this after the LSTM experiment in Kaggle. It preserves the notebook's finite-window filtering and aligns each prediction with its target timestamp.

```python
import joblib
from pathlib import Path

output_dir = Path("/kaggle/working")

def valid_target_times(X, y, timestamps, sequence_length, forecast_horizon=6):
    valid_times = []
    for i in range(sequence_length, len(X) - forecast_horizon + 1):
        window = X.iloc[i-sequence_length:i].values
        target_index = i + forecast_horizon - 1
        target = y.iloc[target_index]
        if np.isfinite(window).all() and np.isfinite(target):
            valid_times.append(timestamps.iloc[target_index])
    return pd.to_datetime(valid_times)

lstm_times = valid_target_times(
    X_test_scaled,
    y_test,
    test_df["Date Time"],
    sequence_length=72,
)

assert len(lstm_times) == len(datasets[72]["test"][1]) == len(predictions["lstm_A_72"])

dashboard_export = pd.DataFrame({
    "timestamp": lstm_times,
    "actual_temperature": datasets[72]["test"][1],
    "lstm_prediction": predictions["lstm_A_72"],
})
dashboard_export.to_csv(output_dir / "predictions_v1.csv", index=False)

models["lstm_A_72"].save(output_dir / "lstm_A_72.keras")
models["lstm_A_72"].save(output_dir / "lstm_A_72.h5")
joblib.dump(fitted_scaler, output_dir / "feature_scaler.pkl")
results_df.to_csv(output_dir / "experiment_results.csv", index=False)
```

The `.keras` file is the recommended Keras format. Keep the `.h5` file as well because the assignment requests a model file in that format.

## Current artifacts

The `artifacts/` folder contains the exported LSTM model, scaler, experiment results, and prediction CSV. The dashboard uses the prediction CSV for its chart and metrics; the model and scaler are retained as deployment artifacts.

