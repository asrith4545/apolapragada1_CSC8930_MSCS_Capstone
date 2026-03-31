import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix

st.set_page_config(page_title="Privacy-Preserving IDS Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

st.title("Privacy-Preserving Intrusion Detection System")
st.caption("Capstone dashboard using UNSW-NB15, Federated Learning, and Differential Privacy")

# ----------------------------
# Helpers
# ----------------------------
def load_csv(filename):
    path = RESULTS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None

def safe_metric_list(df):
    possible = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    return [c for c in possible if c in df.columns]

# ----------------------------
# Load final result files
# ----------------------------
baselines = load_csv("baselines.csv")
fl_iid_final = load_csv("fl_iid_final.csv")
fl_noniid_final = load_csv("fl_noniid_final.csv")
fl_dp_final = load_csv("fl_dp_final.csv")
fl_dp_noniid_final = load_csv("fl_dp_noniid_final.csv")

final_frames = [df for df in [baselines, fl_iid_final, fl_noniid_final, fl_dp_final, fl_dp_noniid_final] if df is not None]

if not final_frames:
    st.error("No final result CSV files were found inside the results folder.")
    st.stop()

final_results = pd.concat(final_frames, ignore_index=True)

# ----------------------------
# Load round-wise files
# ----------------------------
round_data_map = {
    "FL IID": load_csv("fl_iid_rounds.csv"),
    "FL non-IID": load_csv("fl_noniid_rounds.csv"),
    "FL+DP IID": load_csv("fl_dp_rounds.csv"),
    "FL+DP non-IID": load_csv("fl_dp_noniid_rounds.csv"),
}

round_data_map = {k: v for k, v in round_data_map.items() if v is not None}

# ----------------------------
# Load prediction files
# ----------------------------
prediction_data_map = {
    "CatBoost": load_csv("catboost_predictions.csv"),
    "NeuralNet": load_csv("neuralnet_predictions.csv"),
    "RandomForest": load_csv("randomforest_predictions.csv"),
    "FL IID": load_csv("fl_iid_predictions.csv"),
    "FL non-IID": load_csv("fl_noniid_predictions.csv"),
    "FL+DP IID": load_csv("fl_dp_iid_predictions.csv"),
    "FL+DP non-IID": load_csv("fl_dp_noniid_predictions.csv"),
}

prediction_data_map = {k: v for k, v in prediction_data_map.items() if v is not None}

# ----------------------------
# Sidebar
# ----------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Final Comparison", "FL Rounds", "ROC Curves", "Threshold Explorer"]
)

# ----------------------------
# Overview
# ----------------------------
if page == "Overview":
    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset", "UNSW-NB15")

    with col2:
        if "accuracy" in final_results.columns:
            best_idx = final_results["accuracy"].idxmax()
            best_model = final_results.loc[best_idx, "model"]
            best_acc = final_results.loc[best_idx, "accuracy"]
            st.metric("Best Model", str(best_model), f"{best_acc:.4f}")
        else:
            st.metric("Best Model", "Unavailable")

    with col3:
        st.metric("Models Compared", str(len(final_results)))

    st.subheader("Available Final Results")
    st.dataframe(final_results, use_container_width=True)

    st.subheader("Summary")
    st.write("""
    This dashboard presents the results of a privacy-preserving intrusion detection system.
    The project compares centralized learning, federated learning under IID and non-IID settings,
    and federated learning with differential privacy.
    """)

# ----------------------------
# Final Comparison
# ----------------------------
elif page == "Final Comparison":
    st.header("Final Model Comparison")

    metrics = safe_metric_list(final_results)
    metric = st.selectbox("Choose metric", metrics)

    st.dataframe(final_results, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(final_results["model"], final_results[metric])
    ax.set_title(f"{metric.upper()} Comparison")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)

# ----------------------------
# FL Rounds
# ----------------------------
elif page == "FL Rounds":
    st.header("Federated Round-wise Analysis")

    if not round_data_map:
        st.warning("No round-wise CSV files found.")
    else:
        setup = st.selectbox("Choose setup", list(round_data_map.keys()))
        rounds_df = round_data_map[setup]
        metrics = safe_metric_list(rounds_df)
        metric = st.selectbox("Choose round-wise metric", metrics)

        st.dataframe(rounds_df, use_container_width=True)

        if "round" in rounds_df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(rounds_df["round"], rounds_df[metric], marker="o")
            ax.set_title(f"{setup} - {metric.upper()} by Round")
            ax.set_xlabel("Round")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("The selected round file does not contain a 'round' column.")

# ----------------------------
# ROC Curves
# ----------------------------
elif page == "ROC Curves":
    st.header("ROC Curve Comparison")

    if not prediction_data_map:
        st.warning("No prediction CSV files found.")
    else:
        selected_models = st.multiselect(
            "Choose models",
            list(prediction_data_map.keys()),
            default=list(prediction_data_map.keys())[:2]
        )

        if selected_models:
            fig, ax = plt.subplots(figsize=(8, 6))

            for model_name in selected_models:
                df = prediction_data_map[model_name]
                if "y_true" in df.columns and "y_prob" in df.columns:
                    fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Select at least one model.")

# ----------------------------
# Threshold Explorer
# ----------------------------
elif page == "Threshold Explorer":
    st.header("Prediction Threshold Explorer")

    if not prediction_data_map:
        st.warning("No prediction CSV files found.")
    else:
        model_name = st.selectbox("Choose model", list(prediction_data_map.keys()))
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)

        df = prediction_data_map[model_name].copy()

        if "y_prob" in df.columns and "y_true" in df.columns:
            df["pred_label"] = (df["y_prob"] >= threshold).astype(int)

            counts = df["pred_label"].value_counts().sort_index()
            cm = confusion_matrix(df["y_true"], df["pred_label"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Normal (0)", int(counts.get(0, 0)))
            with col2:
                st.metric("Predicted Attack (1)", int(counts.get(1, 0)))

            st.subheader("Confusion Matrix")
            st.write(cm)

            st.subheader("Sample Predictions")
            st.dataframe(df.head(20), use_container_width=True)
        else:
            st.warning("Selected prediction file does not contain y_true and y_prob columns.")
            