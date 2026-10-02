"""
app.py
------
Streamlit web app for the Iris KMeans clustering project.

Run locally:   streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from ml_pipeline import (
    DATA_PATH,
    DEFAULT_K,
    FEATURES,
    MODEL_PATH,
    TARGET,
    assign_clusters,
    elbow_analysis,
    load_artifacts,
    load_data,
    predict_cluster,
    save_artifacts,
    train_model,
)

st.set_page_config(
    page_title="Iris KMeans Clustering",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#6C63FF", "#FF6584", "#2EC4B6", "#FFB703", "#8D99AE", "#EF476F"]
FEATURE_LABELS = {
    "sepal_length": "Sepal Length (cm)",
    "sepal_width": "Sepal Width (cm)",
    "petal_length": "Petal Length (cm)",
    "petal_width": "Petal Width (cm)",
}

st.markdown(
    """
    <style>
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #6C63FF 0%, #FF6584 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; color: white; }
    .hero p { margin: 0.4rem 0 0 0; font-size: 1.05rem; opacity: 0.95; color: white; }
    .result-card {
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        border: 2px solid #6C63FF;
        background: rgba(108, 99, 255, 0.10);
        margin: 0.6rem 0 1rem 0;
    }
    .result-card h2 { margin: 0; }
    .result-card p { margin: 0.3rem 0 0 0; opacity: 0.8; }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 0.8rem 1rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_data(DATA_PATH)


@st.cache_resource(show_spinner="Preparing KMeans model...")
def get_artifacts(k: int) -> dict:
    """Load the saved model for the default k, otherwise train on the fly.

    Model files (*.pkl) are git-ignored, so on Streamlit Cloud the model is
    trained automatically the first time the app starts.
    """
    if k == DEFAULT_K and MODEL_PATH.exists():
        try:
            artifacts = load_artifacts()
            if artifacts["metrics"]["n_clusters"] == k:
                return artifacts
        except Exception:
            pass  # fall through and retrain

    artifacts = train_model(get_data(), n_clusters=k)
    if k == DEFAULT_K:
        try:
            save_artifacts(artifacts)
        except OSError:
            pass  # read-only filesystem: keep the in-memory model
    return artifacts


@st.cache_data(show_spinner=False)
def get_elbow() -> pd.DataFrame:
    return elbow_analysis(get_data(), k_max=8)


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def cluster_color_map(artifacts: dict) -> dict:
    names = [artifacts["cluster_names"][c] for c in sorted(artifacts["cluster_names"])]
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(names)}


def plot_pca_clusters(df, artifacts, colors, user_point=None):
    pipeline, pca = artifacts["pipeline"], artifacts["pca"]
    scaler, kmeans = pipeline.named_steps["scaler"], pipeline.named_steps["kmeans"]

    coords = pca.transform(scaler.transform(df[FEATURES]))
    plot_df = pd.DataFrame(
        {"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": df["cluster_name"].values}
    )
    centers = pca.transform(kmeans.cluster_centers_)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=plot_df, x="PC1", y="PC2", hue="Cluster", palette=colors,
        s=110, edgecolor="white", linewidth=0.8, ax=ax,
    )
    ax.scatter(
        centers[:, 0], centers[:, 1], marker="X", s=260, c="black",
        edgecolor="white", linewidth=1.5, label="Centroids", zorder=5,
    )
    if user_point is not None:
        up = pca.transform(scaler.transform(user_point["row"]))[0]
        ax.scatter(
            up[0], up[1], marker="*", s=600, c="gold", edgecolor="black",
            linewidth=1.5, label="Your flower", zorder=6,
        )
    ax.set_title("KMeans clusters (PCA 2D projection)", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig


def plot_feature_scatter(df, artifacts, colors, x_feat, y_feat, user_point=None):
    pipeline = artifacts["pipeline"]
    scaler, kmeans = pipeline.named_steps["scaler"], pipeline.named_steps["kmeans"]
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=df, x=x_feat, y=y_feat, hue="cluster_name", palette=colors,
        s=110, edgecolor="white", linewidth=0.8, ax=ax,
    )
    ax.scatter(
        centers[x_feat], centers[y_feat], marker="X", s=260, c="black",
        edgecolor="white", linewidth=1.5, label="Centroids", zorder=5,
    )
    if user_point is not None:
        ax.scatter(
            user_point["row"][x_feat], user_point["row"][y_feat], marker="*",
            s=600, c="gold", edgecolor="black", linewidth=1.5, label="Your flower", zorder=6,
        )
    ax.set_title(
        f"{FEATURE_LABELS[x_feat]} vs {FEATURE_LABELS[y_feat]}", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel(FEATURE_LABELS[x_feat])
    ax.set_ylabel(FEATURE_LABELS[y_feat])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [l if l != "cluster_name" else "Cluster" for l in labels], loc="best")
    fig.tight_layout()
    return fig


def plot_elbow(elbow_df, current_k):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(elbow_df["k"], elbow_df["inertia"], marker="o", color=PALETTE[0], linewidth=2.5)
    axes[0].axvline(current_k, color=PALETTE[1], linestyle="--", label=f"k = {current_k}")
    axes[0].set_title("Elbow method (inertia)", fontweight="bold")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].legend()

    axes[1].plot(elbow_df["k"], elbow_df["silhouette"], marker="s", color=PALETTE[2], linewidth=2.5)
    axes[1].axvline(current_k, color=PALETTE[1], linestyle="--", label=f"k = {current_k}")
    axes[1].set_title("Silhouette score", fontweight="bold")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette")
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_distances(result, artifacts, colors):
    names = [artifacts["cluster_names"][c] for c in range(len(result["distances"]))]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.barh(names, result["distances"], color=[colors[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Distance to centroid (standardized units)")
    ax.set_title("Closer = better match", fontweight="bold")
    for bar, value in zip(bars, result["distances"]):
        ax.text(value + 0.03, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
    ax.set_xlim(0, max(result["distances"]) * 1.2)
    fig.tight_layout()
    return fig


def plot_profile_heatmap(df):
    profile = df.groupby("cluster_name")[FEATURES].mean().rename(columns=FEATURE_LABELS)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    sns.heatmap(profile, annot=True, fmt=".2f", cmap="viridis", cbar=True, ax=ax)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("Average feature values per cluster", fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
data = get_data()

st.sidebar.title("🌸 Iris Flower Input")
st.sidebar.caption("Set the measurements of an Iris flower, then press **Predict cluster**.")

inputs = {}
for feature in FEATURES:
    low = float(data[feature].min())
    high = float(data[feature].max())
    pad = (high - low) * 0.25
    inputs[feature] = st.sidebar.slider(
        FEATURE_LABELS[feature],
        min_value=round(max(0.1, low - pad), 1),
        max_value=round(high + pad, 1),
        value=round(float(data[feature].mean()), 1),
        step=0.1,
    )

predict_clicked = st.sidebar.button("🔮 Predict cluster", type="primary")

st.sidebar.divider()
with st.sidebar.expander("⚙️ Advanced settings"):
    k = st.slider("Number of clusters (k)", min_value=2, max_value=6, value=DEFAULT_K)
    st.caption("The classic Iris dataset has 3 species, so k = 3 is the default.")

artifacts = get_artifacts(k)
colors = cluster_color_map(artifacts)

data = data.copy()
data["cluster"] = assign_clusters(artifacts, data)
data["cluster_name"] = data["cluster"].map(artifacts["cluster_names"])

# Keep the last prediction across reruns (e.g. when switching tabs / plot options)
if predict_clicked:
    st.session_state["prediction"] = {
        **predict_cluster(artifacts, **inputs),
        "k": k,
    }
prediction = st.session_state.get("prediction")
if prediction is not None and prediction["k"] != k:
    prediction = None  # model changed, old prediction no longer valid


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
        <h1>🌸 Iris KMeans Clustering</h1>
        <p>Unsupervised machine learning: group Iris flowers into clusters using only their
        measurements and see where a new flower belongs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metrics = artifacts["metrics"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Samples", metrics["n_samples"])
m2.metric("Clusters (k)", metrics["n_clusters"])
m3.metric("Silhouette score", f"{metrics['silhouette']:.3f}")
m4.metric("Inertia", f"{metrics['inertia']:.2f}")

tab_predict, tab_clusters, tab_analysis, tab_data = st.tabs(
    ["🔮 Prediction", "📊 Cluster Visualization", "📈 Model Analysis", "🗂️ Dataset"]
)

with tab_predict:
    left, right = st.columns([1, 1.2])

    with left:
        st.subheader("Your flower")
        st.dataframe(
            pd.DataFrame(
                {"Feature": [FEATURE_LABELS[f] for f in FEATURES],
                 "Value": [inputs[f] for f in FEATURES]}
            ),
            hide_index=True,
        )

        if prediction is None:
            st.info("👈 Adjust the sliders in the sidebar and press **Predict cluster**.")
        else:
            st.markdown(
                f"""
                <div class="result-card">
                    <p>Predicted cluster</p>
                    <h2>{prediction['name']}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
            nearest = float(np.min(prediction["distances"]))
            st.metric("Distance to nearest centroid", f"{nearest:.2f}")
            used = prediction["row"].iloc[0]
            if any(abs(used[f] - inputs[f]) > 1e-9 for f in FEATURES):
                st.caption("ℹ️ Sliders changed since this prediction. Press the button to update.")

    with right:
        st.subheader("Where it lands")
        if prediction is None:
            fig = plot_pca_clusters(data, artifacts, colors)
        else:
            fig = plot_pca_clusters(data, artifacts, colors, prediction)
        st.pyplot(fig)
        plt.close(fig)

    if prediction is not None:
        st.subheader("Distance to every cluster centroid")
        fig = plot_distances(prediction, artifacts, colors)
        st.pyplot(fig)
        plt.close(fig)

with tab_clusters:
    st.subheader("Explore the clusters")
    c1, c2 = st.columns(2)
    x_feat = c1.selectbox(
        "X axis", FEATURES, index=2, format_func=lambda f: FEATURE_LABELS[f]
    )
    y_feat = c2.selectbox(
        "Y axis", FEATURES, index=3, format_func=lambda f: FEATURE_LABELS[f]
    )
    fig = plot_feature_scatter(data, artifacts, colors, x_feat, y_feat, prediction)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Cluster sizes")
    sizes = data["cluster_name"].value_counts().rename_axis("Cluster").reset_index(name="Flowers")
    st.bar_chart(sizes, x="Cluster", y="Flowers")

with tab_analysis:
    st.subheader("Choosing k: elbow & silhouette")
    fig = plot_elbow(get_elbow(), k)
    st.pyplot(fig)
    plt.close(fig)

    a1, a2 = st.columns(2)
    with a1:
        st.subheader("Cluster profiles")
        fig = plot_profile_heatmap(data)
        st.pyplot(fig)
        plt.close(fig)
    with a2:
        if TARGET in data.columns:
            st.subheader("Clusters vs. real species")
            crosstab = pd.crosstab(data["cluster_name"], data[TARGET])
            fig, ax = plt.subplots(figsize=(6, 3.4))
            sns.heatmap(crosstab, annot=True, fmt="d", cmap="mako_r", cbar=False, ax=ax)
            ax.set_xlabel("Actual species")
            ax.set_ylabel("")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            if "purity" in metrics:
                st.caption(
                    f"Cluster purity: **{metrics['purity']:.0%}**. Species labels were never "
                    "used for training; they are only used here to check the result."
                )

with tab_data:
    st.subheader("Dataset with assigned clusters")
    st.dataframe(data.drop(columns=["cluster"]), hide_index=True)
    st.download_button(
        "⬇️ Download clustered data (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="iris_clustered.csv",
        mime="text/csv",
    )
    st.subheader("Summary statistics")
    st.dataframe(data[FEATURES].describe().round(2))

st.divider()
st.caption("Built with Streamlit, scikit-learn, pandas, matplotlib and seaborn.")
