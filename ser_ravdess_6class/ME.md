# Speech Emotion Recognition — Demo

This repository contains a Streamlit app to demo a speech emotion recognition model trained on RAVDESS (6 classes).

How to run locally

1. Create a virtual environment and install requirements:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the app:

```powershell
streamlit run app.py
```

Deploying to Streamlit Community Cloud

- Push this repository to GitHub.
- Sign in to Streamlit Community Cloud and create a new app from this repo (select `app.py`).

Notes
- Model files are placed in `saved_models/`. Keep them in the repo for the demo.
- If you hit memory limits on Streamlit Cloud, use the smaller TFLite model in `saved_models/` and update `app.py` to load it.

---
Generated automatically to prepare deploy.# SER on RAVDESS (6 classes) — Single Notebook
This folder contains **one notebook**: `ser_ravdess_6class.ipynb` implementing the full pipeline.

## Classes
['angry','happy','sad','fearful','disgust','neutral']

## How to use
1. Ensure you have Python 3.10+ and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place/download **Audio_Speech_Actors_01-24.zip** and update the path in the notebook if needed.
   - In this environment it's already at: `/mnt/data/Audio_Speech_Actors_01-24.zip`
3. Open the notebook and run cells in order.

## Outputs
- Prints **TEST ACCURACY** to terminal
- Shows a confusion matrix
- Saves models in `runs/` including `.tflite`

## Notes
- The notebook caches features in `data/processed/` to speed up re-runs.
