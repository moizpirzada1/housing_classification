# Housing Price Classifier (Logistic Regression)

Predicts whether a California district's median house value is **High** or **Low**
(above / below the dataset median, ~$179,700). Two UIs: Gradio and Flask.

## Run locally (Windows)
```
cd housing_project
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py        # creates models/housing_logistic_model.pkl + model_info.json
python app_gradio.py         # http://127.0.0.1:7860
python app_flask.py          # http://127.0.0.1:5000  (use a second terminal)
```
Run `train_model.py` on your own machine so the saved model matches your scikit-learn version.

## Deployment files
- Gradio (Hugging Face Spaces): `app_gradio.py` (rename to `app.py`), `predict_utils.py`, `requirements.txt`, `models/`
- Flask (Render): `app_flask.py`, `predict_utils.py`, `requirements.txt`, `templates/`, `models/`
  - Build: `pip install -r requirements.txt`  Start: `gunicorn app_flask:app`
