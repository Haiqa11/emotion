# 🎤 Speech Emotion Recognition - Web App

A web application for real-time speech emotion recognition using deep learning.

## 🚀 Quick Deploy (Free)

### Deploy to Streamlit Cloud (Recommended)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file: `app.py`
   - Click "Deploy"

3. **Share the URL** with your supervisor!

## 📁 Project Structure

```
ser_ravdess_6class/
├── app.py                 # Streamlit web app
├── requirements.txt       # Python dependencies
├── saved_models/
│   └── ser_baseline.h5   # Trained model (must be included)
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## 🔧 Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.16.1
- Streamlit 1.28+
- See `requirements.txt` for full list

## 🎯 Features

- Upload audio files (WAV, MP3, etc.)
- Real-time emotion prediction
- Confidence scores for all emotions
- Clean, user-friendly interface

## 📖 For Detailed Deployment Instructions

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for step-by-step instructions for multiple platforms.

