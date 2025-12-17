# ⚡ Quick Start Guide - Deploy in 5 Minutes

## 🎯 Fastest Way to Deploy (Streamlit Cloud)

### Step 1: Test Locally (Optional but Recommended)

```bash
# Navigate to project folder
cd "F:\emotion project\emotion_try (1)\emotion_try\ser_ravdess_6class"

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

If it works locally, you're ready to deploy!

### Step 2: Upload to GitHub

**Option A: Using GitHub Web Interface (Easiest)**

1. Go to [github.com](https://github.com) and sign in
2. Click "+" → "New repository"
3. Name: `speech-emotion-recognition`
4. Make it **Public**
5. Click "Create repository"
6. Click "uploading an existing file"
7. Upload these files/folders:
   - `app.py`
   - `requirements.txt`
   - `saved_models/` folder (with `ser_baseline.h5`)
   - `.streamlit/` folder (if exists)
   - `.gitignore`
8. Click "Commit changes"

**Option B: Using Git**

```bash
cd "F:\emotion project\emotion_try (1)\emotion_try\ser_ravdess_6class"
git init
git add app.py requirements.txt saved_models/ .streamlit/ .gitignore
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/speech-emotion-recognition.git
git push -u origin main
```

### Step 3: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `speech-emotion-recognition`
5. Main file path: `app.py`
6. Click "Deploy"

### Step 4: Wait & Share

- Wait 5-10 minutes for deployment
- Copy your app URL (e.g., `https://speech-emotion-recognition.streamlit.app`)
- Share with your supervisor! 🎉

---

## ✅ Checklist Before Deploying

- [ ] `app.py` exists
- [ ] `requirements.txt` exists
- [ ] `saved_models/ser_baseline.h5` exists (model file)
- [ ] Tested locally (optional but recommended)
- [ ] GitHub repository is Public
- [ ] All files uploaded to GitHub

---

## 🆘 Troubleshooting

**"Model not found" error:**
- Make sure `saved_models/ser_baseline.h5` is in your GitHub repo
- Check the file path in GitHub matches the code

**Build fails:**
- Check `requirements.txt` has all dependencies
- Ensure Python 3.8+ compatibility

**App crashes:**
- Check deployment logs in Streamlit Cloud
- Verify model file size is under 1GB

---

## 📞 Need Help?

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed instructions.

