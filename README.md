# MBG Opinion Pulse — IndoBERTweet v1

A dark, artifact-backed Streamlit dashboard for the IndoBERTweet MBG public-opinion classifier. It explores the Kaggle test predictions, confidence profiles, sentiment mix, and held-out validation quality.

The dashboard deliberately does not load the 442 MB model weights during startup. This keeps Streamlit Cloud lightweight and avoids presenting hidden Kaggle test labels as measured outcomes.

## Deployment

The v1 app is deployed at [indobertweet-mbg-v1.streamlit.app](https://indobertweet-mbg-v1.streamlit.app/). It is an artifact-backed analysis dashboard; live comment classification is provided by the separate v2 app.

## Run locally

```powershell
cd D:\dev\mbc-week5-deployment\streamlit_v1
& .\.venv313\Scripts\python.exe -m streamlit run app.py --server.port 8504
```

Open `http://localhost:8504`.

## Bundled artifacts

The app reads these files from `artifacts/` by default:

```text
predictions_test.csv
predictions_validation.csv
classification_report.csv
confusion_matrix.csv
label_mapping.json
model_metadata.json
submission.csv
```

You can also upload `indobertweet_mbg_v1.zip` or an individual `predictions_test.csv` from the sidebar. Uploaded files replace the bundled data for that session.

The IndoBERTweet model is a Hugging Face transformer bundle (`model.safetensors` plus tokenizer/configuration files), not a Keras `.h5` model. This v1 dashboard intentionally does not load it.

## Prediction schema

The required test prediction columns are:

```text
id, comment, predicted_label,
probability_Negative, probability_Neutral, probability_Positive
```

The validation file additionally requires `label`.

## Important boundary

The Kaggle competition test labels are hidden. Therefore the test distribution and confidence views are model outputs, not a public-opinion poll. Validation metrics are calculated only from the labeled hold-out split. Live inference can be added later with a separate model-serving deployment.
