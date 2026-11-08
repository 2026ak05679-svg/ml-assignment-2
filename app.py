"""
Dry Bean Classifier - Streamlit app (ML Assignment 2, M.Tech AIML, BITS Pilani WILP)

Loads the five models trained in Dry_Bean_Classification.ipynb (saved in model/)
and evaluates them on an uploaded test CSV. Nothing is retrained here.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"
TARGET = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbor": "k_nearest_neighbor.pkl",
    "Naive Bayes (Gaussian)": "naive_bayes_gaussian.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}
    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    return models, encoder


def feature_columns(model):
    # The pipelines were fitted on a DataFrame, so they remember the column order
    return list(model.feature_names_in_)


def evaluate(model, X, y_true, n_classes):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted",
                            labels=list(range(n_classes)))
    except ValueError:      # e.g. the uploaded file does not contain every class
        auc = np.nan
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_pred


models, encoder = load_artifacts()
CLASS_NAMES = list(encoder.classes_)
FEATURES = feature_columns(next(iter(models.values())))

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload a test CSV", type="csv",
                                help=f"Needs the {len(FEATURES)} feature columns and a '{TARGET}' column.")
    use_sample = st.checkbox("Use the bundled test_data.csv instead", value=uploaded is None)
    if SAMPLE_CSV.exists():
        st.download_button("Download test_data.csv", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))

# ---------------------------------------------------------------- header
st.title("🫘 Dry Bean Variety Classifier")
st.caption("Five classifiers trained on the UCI Dry Bean dataset (13,611 beans, 16 shape "
           "features, 7 varieties). The models are pre-trained; this app only evaluates them "
           "on the test data you provide.")

if uploaded is not None and not use_sample:
    df = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample and SAMPLE_CSV.exists():
    df = pd.read_csv(SAMPLE_CSV)
    source = "test_data.csv (20% hold-out split)"
else:
    st.info("Upload a CSV in the sidebar, or tick **Use the bundled test_data.csv**.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The file is missing these columns: {', '.join(missing)}")
    st.stop()

unknown = sorted(set(df[TARGET].astype(str)) - set(CLASS_NAMES))
if unknown:
    st.error(f"Unknown class labels in '{TARGET}': {', '.join(unknown)}. "
             f"Expected: {', '.join(CLASS_NAMES)}")
    st.stop()

df = df.dropna(subset=FEATURES + [TARGET])
X = df[FEATURES]
y_true = encoder.transform(df[TARGET].astype(str))

st.write(f"**Data:** {source} · {len(df):,} rows · {df[TARGET].nunique()} classes present")
with st.expander("Preview the data"):
    st.dataframe(df.head(20))

tab_single, tab_compare = st.tabs([f"📊 {model_name}", "🏆 Compare all models"])

# ---------------------------------------------------------------- one model
with tab_single:
    metrics, y_pred = evaluate(models[model_name], X, y_true, len(CLASS_NAMES))

    cols = st.columns(6)
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, "N/A" if np.isnan(value) else f"{value:.4f}")
    if np.isnan(metrics["AUC"]):
        st.caption("AUC needs every class to appear in the uploaded file, so it is shown as N/A.")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
        fig, ax = plt.subplots(figsize=(6.2, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted variety")
        ax.set_ylabel("Actual variety")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Classification report")
        report = classification_report(y_true, y_pred, labels=range(len(CLASS_NAMES)),
                                       target_names=CLASS_NAMES, output_dict=True,
                                       zero_division=0)
        report_df = pd.DataFrame(report).T
        report_df.loc["accuracy", ["precision", "recall"]] = np.nan
        report_df.loc["accuracy", "support"] = len(y_true)
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(report_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                             "f1-score": "{:.4f}", "support": "{:d}"}, na_rep=""))

    results = df.copy()
    results["Predicted"] = encoder.inverse_transform(y_pred)
    results["Correct"] = results["Predicted"] == results[TARGET]
    st.download_button(f"Download {model_name} predictions (CSV)",
                       results.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].replace('.pkl', '')}.csv",
                       mime="text/csv")

# ---------------------------------------------------------------- all models
with tab_compare:
    rows = {name: evaluate(m, X, y_true, len(CLASS_NAMES))[0] for name, m in models.items()}
    table = pd.DataFrame(rows).T
    table.index.name = "Model"
    st.dataframe(table.style.format("{:.4f}", na_rep="N/A").highlight_max(axis=0, color="#cfe3fa"))
    best = table["Accuracy"].idxmax()
    st.success(f"Best accuracy on this data: **{best}** ({table.loc[best, 'Accuracy']:.4f})")

    fig, ax = plt.subplots(figsize=(10, 4))
    table.drop(columns="AUC" if table["AUC"].isna().all() else []).plot.bar(ax=ax, width=0.8)
    ax.set_ylim(max(0.0, np.nanmin(table.values) - 0.05), 1.0)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
