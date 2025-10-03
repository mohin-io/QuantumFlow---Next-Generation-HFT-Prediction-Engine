# 🚀 Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

Follow these steps to deploy your QuantumFlow Trading Dashboard to Streamlit Cloud.

---

## Prerequisites

✅ GitHub account (free)
✅ Streamlit Cloud account (free) - will create during deployment
✅ Your code already pushed to GitHub

---

## Step-by-Step Deployment

### Step 1: Commit and Push Files to GitHub

First, add the new deployment files to your repository:

```bash
# Add the new files
git add run_trading_dashboard.py
git add src/visualization/app_utils.py
git add tests/test_app_utils.py
git add tests/test_streamlit_app.py
git add requirements-streamlit.txt
git add .streamlit/config.toml
git add packages.txt
git add STREAMLIT_APP_README.md
git add STREAMLIT_CLOUD_DEPLOYMENT.md

# Commit the changes
git commit -m "Add Streamlit trading dashboard with comprehensive testing"

# Push to GitHub
git push origin master
```

**Note**: If you get authentication errors, you may need to set up a GitHub Personal Access Token.

---

### Step 2: Sign Up for Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Click **"Sign up"** or **"Get started"**
3. Sign in with your **GitHub account**
4. Authorize Streamlit to access your repositories

---

### Step 3: Deploy Your App

1. Once logged in to Streamlit Cloud, click **"New app"**

2. Fill in the deployment form:
   - **Repository**: `mohin-io/QuantumFlow---Next-Generation-HFT-Prediction-Engine`
   - **Branch**: `master` (or `main`)
   - **Main file path**: `run_trading_dashboard.py`

3. Click **"Advanced settings"** (optional):
   - **Python version**: 3.10 or 3.11
   - **Secrets**: (leave empty for now)

4. Click **"Deploy!"**

---

### Step 4: Wait for Deployment

Streamlit Cloud will:
1. Clone your repository
2. Install dependencies from `requirements-streamlit.txt`
3. Build and launch your app

This typically takes **2-5 minutes**.

You'll see logs showing:
```
Cloning repository...
Installing requirements...
Building app...
App deployed successfully!
```

---

### Step 5: Access Your Live App

Once deployed, you'll get a URL like:
```
https://quantumflow-trading-dashboard.streamlit.app
```

Or something similar based on your app name.

**Share this URL** with anyone - the app is now publicly accessible!

---

## Alternative: Use requirements.txt (Instead of requirements-streamlit.txt)

If Streamlit Cloud doesn't find `requirements-streamlit.txt`, it will use `requirements.txt`.

To ensure it uses the minimal dependencies, you can rename:

```bash
# Backup original requirements
mv requirements.txt requirements-full.txt

# Use streamlit requirements
cp requirements-streamlit.txt requirements.txt

# Commit and push
git add requirements.txt requirements-full.txt
git commit -m "Use minimal requirements for Streamlit Cloud"
git push
```

---

## Troubleshooting

### Issue: "Requirements installation failed"

**Solution**: The full `requirements.txt` has too many dependencies. Make sure you're using `requirements-streamlit.txt`:

1. In Streamlit Cloud dashboard, go to **Settings** → **Advanced**
2. Change **Requirements file** to `requirements-streamlit.txt`
3. Click **Save** and **Reboot app**

### Issue: "Module not found error"

**Solution**: Add the missing module to `requirements-streamlit.txt`:

```txt
# Add the missing package
missing-package>=1.0.0
```

Then commit and push:
```bash
git add requirements-streamlit.txt
git commit -m "Add missing dependency"
git push
```

Streamlit Cloud will auto-redeploy on push.

### Issue: "App not loading / showing errors"

**Solution**: Check the logs in Streamlit Cloud:
1. Go to your app dashboard
2. Click **"Manage app"** → **"Logs"**
3. Look for error messages
4. Fix the issue locally and push again

### Issue: "Git authentication failed"

**Solution**: Create a GitHub Personal Access Token:
1. Go to GitHub → Settings → Developer Settings → Personal Access Tokens
2. Generate new token (classic)
3. Select scopes: `repo`, `workflow`
4. Copy the token
5. Use it as your password when pushing:
   ```bash
   git push
   Username: your-username
   Password: <paste-token-here>
   ```

---

## App Management

### Updating Your App

Any time you push to GitHub, Streamlit Cloud will **auto-redeploy**:

```bash
# Make changes to your app
# ... edit files ...

# Commit and push
git add .
git commit -m "Update dashboard features"
git push

# Streamlit Cloud detects the push and redeploys automatically!
```

### Viewing App Analytics

In the Streamlit Cloud dashboard:
- **Viewers**: See who's using your app
- **Usage**: Monitor resource consumption
- **Logs**: Debug issues in real-time

### Managing Settings

Click **"Settings"** in your app dashboard to:
- Change Python version
- Add secrets (API keys, etc.)
- Configure custom domains
- Set resource limits

---

## Advanced: Using Secrets

If you need to connect to real APIs in the future:

1. In Streamlit Cloud, go to **Settings** → **Secrets**
2. Add your secrets in TOML format:
   ```toml
   [api]
   url = "https://api.example.com"
   key = "your-secret-key"
   ```

3. Access in your app:
   ```python
   import streamlit as st

   api_url = st.secrets["api"]["url"]
   api_key = st.secrets["api"]["key"]
   ```

---

## Current App Features (Live Demo Ready!)

Your deployed app will show:

✅ **Order Book Tab**: Real-time depth visualization
✅ **Trading Signals Tab**: AI predictions with confidence
✅ **Performance Tab**: Sharpe ratio, drawdown, PnL curves
✅ **Features Tab**: SHAP importance, correlations
✅ **Logs Tab**: System monitoring

All using **simulated data** - perfect for demonstrations!

---

## Next Steps After Deployment

### Immediate (< 5 minutes)
1. ✅ Verify app loads at your Streamlit Cloud URL
2. ✅ Test all 5 tabs work correctly
3. ✅ Try sidebar controls (exchange, symbol, model)
4. ✅ Share URL with team/stakeholders

### Short-term (1-2 weeks)
5. Connect to real API backend (when ready)
6. Add authentication if needed
7. Configure custom domain (optional)
8. Monitor usage and performance

---

## Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Streamlit Cloud**: https://share.streamlit.io
- **Community Forum**: https://discuss.streamlit.io
- **Your App README**: [STREAMLIT_APP_README.md](STREAMLIT_APP_README.md)

---

## Support

If you encounter issues:
1. Check logs in Streamlit Cloud dashboard
2. Review [STREAMLIT_APP_README.md](STREAMLIT_APP_README.md)
3. Run `python verify_deployment.py` locally
4. Check Streamlit Community Forum

---

## Summary

**What you're deploying:**
- ✅ Professional trading dashboard
- ✅ 5 interactive tabs
- ✅ 66 tests (all passing)
- ✅ Production-ready code
- ✅ Minimal dependencies (fast deployment)

**Deployment time:** ~2-5 minutes
**Cost:** FREE on Streamlit Cloud
**Updates:** Auto-deploy on git push

---

**Ready to deploy? Follow the steps above! 🚀**
