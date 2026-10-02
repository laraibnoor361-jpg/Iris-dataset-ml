# 🌸 Iris KMeans Clustering: Unsupervised ML Web App

An interactive **machine learning web application** that groups Iris flowers into clusters with
**KMeans** (unsupervised learning) and lets you predict the cluster of a new flower from its
measurements.

## Why this is an AI/ML web application

- **Unsupervised learning:** KMeans finds structure in the data without using species labels.
- **Full ML pipeline:** preprocessing, feature scaling (`StandardScaler`), model training,
  evaluation (silhouette score, inertia, elbow method) and saved model artifacts.
- **Live inference:** enter sepal and petal measurements in the sidebar and get an instant
  predicted cluster plus distances to every centroid.
- **Interactive visualizations:** PCA cluster plot, feature scatter plots, elbow chart,
  cluster profiles and a cluster vs. species comparison.
- **Deployable:** the app trains the model automatically on first start, so it runs on
  Streamlit Community Cloud without committing any model file.

## Project structure

```
iris-kmeans-clustering-ml/
├── app.py              # Streamlit web app
├── ml_pipeline.py      # preprocessing, scaling, KMeans training, prediction
├── train_ml_model.py   # trains and saves the model (joblib)
├── test_ml_model.py    # loads the model and predicts new flowers
├── dataset.csv         # sample Iris data (18 rows)
├── requirements.txt
├── .gitignore
└── README.md
```

## Run locally

```bash
git clone https://github.com/<your-username>/iris-kmeans-clustering-ml.git
cd iris-kmeans-clustering-ml

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python train_ml_model.py        # train and save the model
python test_ml_model.py         # test predictions
streamlit run app.py            # launch the web app
```

### Using the full 150-row Iris dataset

`dataset.csv` is a small sample. To use the complete dataset, replace the file with your
`Iris.csv`. Column names such as `SepalLengthCm` / `Species` and the extra `Id` column are
handled automatically.

## Push to GitHub

```bash
git init
git add .
git commit -m "Iris KMeans clustering ML project"
git branch -M main
git remote add origin https://github.com/<your-username>/iris-kmeans-clustering-ml.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud

1. Make sure the repository is on GitHub and contains `app.py` and `requirements.txt`.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** (or **New app**) and choose **Deploy a public app from GitHub**.
4. Select the repository `iris-kmeans-clustering-ml`, branch `main`, and main file path `app.py`.
5. Click **Deploy**. Streamlit installs the packages from `requirements.txt` and starts the app.
6. When the build finishes, share your public `*.streamlit.app` link.

> The `.pkl` model file is git-ignored on purpose. The app trains the model on first launch
> and reuses it afterwards.

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib.
