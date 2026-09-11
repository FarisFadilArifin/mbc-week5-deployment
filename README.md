# MBG Opinion Studio — IndoBERTweet v2

Indonesian-first Streamlit application for the regularized five-fold IndoBERTweet MBG classifier. Users can type a comment, classify it with the best fold model, inspect class probabilities, and explore the v2 out-of-fold/test exports.

The v2 test set is unlabeled. Its distribution is therefore a model-output view, not a measured opinion poll.

## Local run

Use Python 3.12 for live inference, matching Streamlit Community Cloud’s default runtime:

```powershell
cd D:\dev\mbc-week5-deployment\streamlit_v1
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.port 8504
```

The app still starts without v2 artifacts or model credentials and shows an actionable setup state.

## V2 artifacts

After running `notebooks/indobertweet_mbg_v2_5fold.ipynb` on Kaggle, place or upload:

```text
predictions_test_v2.csv
predictions_oof_v2.csv
fold_metrics_v2.csv
classification_report_v2.csv
confusion_matrix_v2.csv
model_metadata_v2.json
label_mapping_v2.json
```

The sidebar accepts the exported v2 ZIP or individual CSV/JSON files. Uploaded data replaces bundled data for that session; missing optional files hide their corresponding panels.

## Live model configuration

For local testing, extract the best-fold model to:

```text
artifacts/indobertweet_model/
```

For Streamlit Cloud, upload that folder to a private Hugging Face model repository and configure these secrets:

```toml
HF_MODEL_ID = "your-account/indobertweet-mbg-v2"
HF_TOKEN = "hf_..."
```

The model is loaded lazily only after the user clicks **Klasifikasikan komentar** and is cached for subsequent predictions. Never commit `HF_TOKEN` or model weights.

## Training notebook

The v2 Kaggle notebook is [notebooks/indobertweet_mbg_v2_5fold.ipynb](notebooks/indobertweet_mbg_v2_5fold.ipynb). It trains five stratified folds with stronger dropout, weight decay, label smoothing, gradient clipping, and early stopping. Live inference uses the best fold; exported test predictions average all five folds.

## Boundary and privacy

Typed comments remain in Streamlit session memory only. They are not written to disk or sent to an analytics service. Predictions are model outputs and should not be treated as moderation decisions or objective truth.
