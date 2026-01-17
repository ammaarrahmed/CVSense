# Kaggle API Setup Guide

This guide will help you set up Kaggle API credentials to download datasets for CVSense.

---

## Quick Setup (Recommended)

### Step 1: Get Your Kaggle API Credentials

1. Go to [Kaggle.com](https://www.kaggle.com) and sign in (create account if needed)
2. Click on your profile picture (top right) → **Account**
3. Scroll down to **API** section
4. Click **"Create New API Token"**
5. This downloads `kaggle.json` to your computer

### Step 2: Choose Your Setup Method

You have **two options**:

---

## Option A: Environment Variables (Recommended for this project)

This keeps credentials in the project folder (but excluded from git).

1. **Create `.env` file** in the project root:
   ```bash
   cd /home/ammaar/CODE/CVSense
   cp .env.example .env
   ```

2. **Open the downloaded `kaggle.json`** file and copy the values:
   ```json
   {
     "username": "your_username",
     "key": "abc123def456..."
   }
   ```

3. **Edit `.env`** and paste your credentials:
   ```bash
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=abc123def456...
   ```

4. **Done!** The notebook will automatically load these credentials.

---

## Option B: System-Wide Setup (Traditional Kaggle method)

This installs credentials for all your Kaggle projects.

1. **Create Kaggle directory:**
   ```bash
   mkdir -p ~/.kaggle
   ```

2. **Move the downloaded `kaggle.json`:**
   ```bash
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   ```

3. **Set proper permissions (Linux/Mac):**
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Done!** All Kaggle tools will now work automatically.

---

## Verification

Test your setup by running this in a terminal:

```bash
kaggle datasets list
```

If you see a list of datasets, you're all set! ✅

---

## Security Notes

⚠️ **NEVER commit your API credentials to git!**

- ✅ `.env` is in `.gitignore` (safe)
- ✅ `kaggle.json` is in `.gitignore` (safe)
- ✅ `.env.example` has no real credentials (safe to commit)
- ❌ Do NOT remove these from `.gitignore`
- ❌ Do NOT share your API key publicly

---

## Troubleshooting

### "401 Unauthorized" Error
- **Cause:** Wrong credentials
- **Fix:** Double-check username and key in `.env` or `kaggle.json`

### "kaggle.json not found"
- **Cause:** File not in correct location
- **Fix:** 
  - Option A: Make sure `.env` exists in project root
  - Option B: Make sure `kaggle.json` is in `~/.kaggle/`

### "Permission denied" (Linux/Mac)
- **Cause:** Wrong file permissions
- **Fix:** `chmod 600 ~/.kaggle/kaggle.json`

### Notebook can't download dataset
- **Cause:** Credentials not loaded
- **Fix:** Restart the kernel and run cells from the beginning

---

## For Team Members Cloning This Project

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd CVSense
   ```

2. **Follow Option A above** to create your own `.env` file

3. **Never commit your `.env`** (it's already in `.gitignore`)

4. **Run the notebook** - it will use your credentials automatically

---

## Need Help?

- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- Check [QUICK_START.md](module_1_data_ingestion/QUICK_START.md)
- Contact Module 1 owner
