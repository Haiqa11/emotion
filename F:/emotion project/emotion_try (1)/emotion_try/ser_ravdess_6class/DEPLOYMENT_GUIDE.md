# 🚀 Free Deployment Guide - Speech Emotion Recognition

This guide will help you deploy your Speech Emotion Recognition app **completely free** using Streamlit Cloud.

## 📋 Prerequisites

1. A GitHub account (free)
2. Your project files ready
3. A trained model file (`ser_baseline.h5`)

## 🎯 Option 1: Streamlit Cloud (Recommended - Easiest & Free)

### Step 1: Prepare Your Project

1. Make sure your project structure looks like this:
```
emotion_try/
└── ser_ravdess_6class/
    ├── app.py                    # Streamlit app
    ├── requirements.txt          # Dependencies
    ├── saved_models/
    │   └── ser_baseline.h5      # Your trained model
    └── .streamlit/
        └── config.toml          # Streamlit config (optional)
```

### Step 2: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon → "New repository"
3. Name it (e.g., `speech-emotion-recognition`)
4. Make it **Public** (required for free Streamlit Cloud)
5. Click "Create repository"

### Step 3: Upload Your Code to GitHub

**Option A: Using GitHub Desktop (Easiest)**
1. Download [GitHub Desktop](https://desktop.github.com/)
2. Clone your repository
3. Copy your project files to the repository folder
4. Commit and push

**Option B: Using Git Command Line**
```bash
cd "F:\emotion project\emotion_try (1)\emotion_try\ser_ravdess_6class"
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

**Option C: Using GitHub Web Interface**
1. Go to your repository on GitHub
2. Click "uploading an existing file"
3. Drag and drop your files
4. Commit changes

### Step 4: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set:
   - **Main file path**: `app.py`
   - **Branch**: `main` (or your default branch)
6. Click "Deploy"

### Step 5: Wait for Deployment

- First deployment takes 5-10 minutes
- You'll see build logs in real-time
- Once done, you'll get a URL like: `https://your-app.streamlit.app`

### Step 6: Share Your App

Your app is now live! Share the URL with your supervisor.

---

## 🎯 Option 2: Hugging Face Spaces (Alternative - Also Free)

### Step 1: Create Hugging Face Account

1. Go to [huggingface.co](https://huggingface.co)
2. Sign up for free account

### Step 2: Create a New Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **Space name**: `speech-emotion-recognition`
   - **SDK**: `Streamlit`
   - **Hardware**: `CPU basic` (free)
   - **Visibility**: `Public`

### Step 3: Upload Files

1. Clone the space repository:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/speech-emotion-recognition
```

2. Copy your files:
   - `app.py`
   - `requirements.txt`
   - `saved_models/ser_baseline.h5`
   - `.streamlit/config.toml` (if exists)

3. Push to Hugging Face:
```bash
cd speech-emotion-recognition
git add .
git commit -m "Add emotion recognition app"
git push
```

### Step 4: Access Your App

Your app will be available at:
`https://huggingface.co/spaces/YOUR_USERNAME/speech-emotion-recognition`

---

## 🎯 Option 3: Render (Alternative - Free Tier Available)

### Step 1: Create Render Account

1. Go to [render.com](https://render.com)
2. Sign up with GitHub

### Step 2: Create New Web Service

1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Configure:
   - **Name**: `speech-emotion-recognition`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Plan**: `Free`

### Step 3: Deploy

1. Click "Create Web Service"
2. Wait for deployment (5-10 minutes)
3. Your app will be at: `https://your-app.onrender.com`

**Note**: Free tier apps sleep after 15 minutes of inactivity.

---

## 📝 Important Notes

### Model File Size

- Streamlit Cloud: Up to 1GB per app (should be fine)
- Hugging Face: Up to 50GB (plenty of space)
- Render: 512MB free tier

If your model is too large:
1. Use the pruned model (`ser_pruned_manual.h5`) - smaller file size
2. Or convert to TFLite format (much smaller)

### Environment Variables (if needed)

If you need to set environment variables:
- **Streamlit**: Add in the app settings
- **Hugging Face**: Add in Space settings → Variables
- **Render**: Add in Environment section

### Troubleshooting

**Issue: Model not found**
- Ensure `saved_models/ser_baseline.h5` is in your repository
- Check the path in `app.py` matches your file structure

**Issue: Build fails**
- Check `requirements.txt` has all dependencies
- Ensure Python version compatibility (3.8+)

**Issue: App crashes**
- Check Streamlit Cloud logs for errors
- Test locally first: `streamlit run app.py`

---

## 🎉 Quick Start (Recommended Path)

1. **Create GitHub repo** → Upload code
2. **Deploy to Streamlit Cloud** → Get URL
3. **Share with supervisor** → Done!

**Total time**: ~15 minutes
**Cost**: $0 (completely free)

---

## 📞 Need Help?

- Streamlit Cloud Docs: https://docs.streamlit.io/streamlit-community-cloud
- Streamlit Forum: https://discuss.streamlit.io
- GitHub Issues: Create an issue in your repo

Good luck with your deployment! 🚀

