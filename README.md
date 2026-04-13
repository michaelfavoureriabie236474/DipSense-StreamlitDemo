# ML Drift Monitor

A Streamlit dashboard for monitoring data drift and model performance for a credit card fraud detection model.

## Project structure

- `dashboard/app.py` - main Streamlit app
- `src/retrain_logic.py` - retraining decision logic
- `Dataset/` - baseline and drift datasets
- `Models/` - trained model, scaler, and baseline metrics
- `requirements.txt` - required Python packages

## How to run locally

```bash
pip install -r requirements.txt
python3 -m streamlit run dashboard/app.py