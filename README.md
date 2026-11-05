# Crop Disease and Pest Detection System

Deep Learning: Custom CNN and MobileNetV2 for crop disease detection.

The app classifies a leaf photo into 22 disease, pest, and healthy classes across four crops (Cashew, Cassava, Maize, Tomato). It shows the prediction with a confidence level, treatment and prevention guidance, a Grad-CAM heatmap, a PDF report, an AI agronomist chat, and an expert directory.

Predictions are screening results, not a confirmed laboratory diagnosis.

## Project structure

```
classifier_app/
  crop_classifier_dash.py      Streamlit app (main file)
  requirements.txt             Python packages
  modules/                     App modules (model loader, predictor, Grad-CAM, PDF, experts, disease database)
  deployment_model/
    cnn/                       crop_model.keras, class_names.json, model_config.json
    mobilenetv2/               crop_model.keras, class_names.json, model_config.json
  .gitignore                   Files that Git must not upload
```

## Models

| Model | Input size | Validation accuracy |
|---|---|---|
| MobileNetV2 | 128 x 128 | 82.5% |
| Custom CNN | 96 x 96 | See deployment_model/cnn/classification_report.txt |

## Run the app on your computer

Open a terminal in the project folder, then run:

```
pip install -r requirements.txt
streamlit run crop_classifier_dash.py
```

The MobileNetV2 file was saved with Keras 2. The first two lines of `crop_classifier_dash.py` set `TF_USE_LEGACY_KERAS` to `1` so that TensorFlow loads it correctly. Keep those lines at the very top of the file.

## Push the project from VS Code to GitHub

Use the VS Code terminal (Terminal, then New Terminal). The terminal must show the project folder. Check it with:

```
Get-Location
```

Type the quotation marks yourself if you copy a command from a chat window, because chat apps replace straight quotes with curly quotes and Git then fails.

### First time only

1. Create an empty repository on github.com. Leave Add a README, Add .gitignore, and Choose a license unchecked.
2. Confirm that Git is installed:

```
git --version
```

3. Run these commands one at a time. Replace the address with your own repository address.

```
git init
git add .
git status
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/kera-na/crop-disease-detection.git
git push -u origin main
```

4. If GitHub opens a sign-in window in your browser, sign in and approve it.

### Every later update

Run these commands after you change or add files:

```
git status
git add .
git commit -m "Describe your change here"
git push
```

Example: after you add the CNN model files, run `git add deployment_model/cnn`, then `git commit -m "Add CNN model"`, then `git push`.

### Use the VS Code buttons instead of commands

1. Click the Source Control icon in the left bar (the branch symbol).
2. Type a commit message in the box.
3. Click Commit, then click Sync Changes.

### Download changes from GitHub

```
git pull
```

### Common problems

| Message | Fix |
|---|---|
| Please tell me who you are | Run `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"`, then commit again |
| src refspec main does not match any | No commit exists yet. Run `git commit -m "Initial commit"`, then push again |
| remote origin already exists | Run `git remote set-url origin https://github.com/kera-na/crop-disease-detection.git` |
| Updates were rejected | Run `git pull --rebase origin main`, then `git push` |
| invalid character in a Python file | A curly quote slipped in. Retype the quotation marks by hand |
| File is larger than 100 MB | GitHub blocks it. Use Git LFS or keep the file out of the repository |

### Keep these files out of GitHub

The `.gitignore` file lists them. Never commit passwords, API keys, or database files.

```
*.pyc
.env
.streamlit/secrets.toml
*.db
```

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub using the commands above. The repository must be public, or Streamlit must have access to your private repositories.
2. Open share.streamlit.io and sign in with GitHub.
3. Click Create app, then select the repository, the branch `main`, and the main file `crop_classifier_dash.py`.
4. Open Advanced settings, choose Python 3.11, and add this secret with your own strong password (type the quotes by hand):

```
ADMIN_PASSWORD = "your-strong-password"
```

5. Click Deploy. Every `git push` to `main` rebuilds the app.

Set Analysis mode to MobileNetV2 in the sidebar if the CNN files are missing. The Compare Both Models mode needs both models.