
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans, DBSCAN, Birch, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #f4f5f7;
}
[data-testid="stSidebar"] {
    background-color: #262832;
}
[data-testid="stMetric"] {
    background-color: #171a22;
    border: 1px solid #303442;
    padding: 18px;
    border-radius: 12px;
}
[data-testid="stMetricLabel"] {
    color: #cbd0da;
}
[data-testid="stMetricValue"] {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Customer Segmentation Dashboard")
st.caption(
    "Customer Personality Analysis using Clustering "
    "for targeted marketing."
)

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def clean_numeric_data(data):
    numeric = data.select_dtypes(include=np.number).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")

    if not numeric.empty:
        numeric = numeric.fillna(numeric.median()).fillna(0)
        numeric = numeric.loc[
            :, numeric.nunique(dropna=False) > 1
        ]

    return numeric


def calculate_metrics(X, labels, model_name):
    labels = np.asarray(labels)

    if model_name == "DBSCAN":
        mask = labels != -1
    else:
        mask = np.ones(len(labels), dtype=bool)

    valid_X = X[mask]
    valid_labels = labels[mask]
    unique = np.unique(valid_labels)

    if model_name == "DBSCAN":
        cluster_count = len(set(labels) - {-1})
        noise_count = int(np.sum(labels == -1))
    else:
        cluster_count = len(unique)
        noise_count = 0

    silhouette = np.nan
    db_index = np.nan
    ch_score = np.nan

    if len(valid_labels) >= 3 and 1 < len(unique) < len(valid_labels):
        try:
            silhouette = silhouette_score(valid_X, valid_labels)
            db_index = davies_bouldin_score(valid_X, valid_labels)
            ch_score = calinski_harabasz_score(valid_X, valid_labels)
        except Exception:
            pass

    return {
        "Model": model_name,
        "Clusters": cluster_count,
        "Noise Points": noise_count,
        "Silhouette Score": round(float(silhouette), 4)
            if np.isfinite(silhouette) else np.nan,
        "Davies-Bouldin Index": round(float(db_index), 4)
            if np.isfinite(db_index) else np.nan,
        "Calinski-Harabasz Score": round(float(ch_score), 4)
            if np.isfinite(ch_score) else np.nan,
    }


def nearest_cluster(X, labels, new_X):
    """Approximate cluster assignment for models without predict()."""
    cluster_ids = [
        label for label in np.unique(labels) if label != -1
    ]

    if not cluster_ids:
        return "Unassigned"

    centers = np.array([
        X[labels == cluster_id].mean(axis=0)
        for cluster_id in cluster_ids
    ])

    distances = np.sum((centers - new_X[0]) ** 2, axis=1)
    return str(cluster_ids[np.argmin(distances)])


# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("Dashboard Filters")

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Dataset",
    type=["csv", "xlsx", "xls"],
)

st.sidebar.divider()
st.sidebar.subheader("Clustering Settings")

k = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=4,
)

eps = st.sidebar.slider(
    "DBSCAN eps",
    min_value=0.1,
    max_value=3.0,
    value=0.8,
    step=0.1,
)

min_samples = st.sidebar.slider(
    "DBSCAN min samples",
    min_value=2,
    max_value=20,
    value=5,
)

birch_threshold = st.sidebar.slider(
    "Birch threshold",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1,
)

# =====================================================
# LOAD DATA
# =====================================================
if uploaded_file is None:
    default_file = BASE_DIR / "segmented_customers.csv"

    if default_file.exists():
        df = pd.read_csv(default_file)
        st.sidebar.caption("Loaded segmented_customers.csv")
    else:
        st.info(
            "Upload your customer CSV or Excel file "
            "using the left sidebar."
        )

        a, b, c = st.columns(3)
        a.metric("Customers in View", "—")
        b.metric("Customer Segments", "—")
        c.metric("Average Income", "—")

        st.markdown("""
        ### Dashboard Features

        - Segment Analysis
        - Feature Explorer
        - Correlation Analysis
        - Model Evaluation
        - Predict New Customers
        """)
        st.stop()
else:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as error:
        st.error(f"Unable to read dataset: {error}")
        st.stop()

if df.empty or len(df) < 3:
    st.error("Dataset must contain at least 3 customer records.")
    st.stop()

# Remove old cluster labels to avoid using them as input features.
df = df.drop(columns=["Segment"], errors="ignore")

id_columns = [
    col for col in df.columns
    if str(col).strip().lower() in {
        "id", "customerid", "customer_id",
        "clientid", "client_id", "row_id"
    }
]

numeric_df = clean_numeric_data(
    df.drop(columns=id_columns, errors="ignore")
)

if numeric_df.shape[1] < 2:
    st.error(
        "At least two usable numeric feature columns are required."
    )
    st.stop()

k = min(k, len(df) - 1)
spectral_neighbors = min(10, len(df) - 1)

scaler = StandardScaler()
X = scaler.fit_transform(numeric_df)

# =====================================================
# FOUR CLUSTERING ALGORITHMS
# =====================================================
models = {
    "K-Means": KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    ),
    "DBSCAN": DBSCAN(
        eps=eps,
        min_samples=min_samples,
    ),
    "Birch": Birch(
        n_clusters=k,
        threshold=birch_threshold,
    ),
    "Spectral Clustering": SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=spectral_neighbors,
        random_state=42,
        assign_labels="kmeans",
    ),
}

labels_by_model = {}
evaluation_rows = []
model_errors = {}

with st.spinner("Running clustering algorithms..."):
    for model_name, model in models.items():
        try:
            labels = model.fit_predict(X)
            labels_by_model[model_name] = labels

            evaluation_rows.append(
                calculate_metrics(X, labels, model_name)
            )
        except Exception as error:
            model_errors[model_name] = str(error)

if not labels_by_model:
    st.error("No clustering model could run on this dataset.")
    for name, error in model_errors.items():
        st.write(f"{name}: {error}")
    st.stop()

if model_errors:
    with st.expander("Models with errors"):
        for name, error in model_errors.items():
            st.warning(f"{name}: {error}")

# =====================================================
# FILTERS AND KPI CARDS
# =====================================================
st.sidebar.divider()
filter_model = st.sidebar.selectbox(
    "Filter Using Model",
    list(labels_by_model.keys()),
)

selected_labels = labels_by_model[filter_model]

df = df.copy()
df["Segment"] = [
    "Noise" if str(label) == "-1"
    else f"Cluster {label}"
    for label in selected_labels
]

segment_options = sorted(df["Segment"].unique())

selected_segments = st.sidebar.multiselect(
    "Select Customer Segments",
    segment_options,
    default=segment_options,
)

mask = df["Segment"].isin(selected_segments)
filtered_df = df.loc[mask].copy()

income_columns = [
    col for col in numeric_df.columns
    if "income" in str(col).lower()
]

k1, k2, k3 = st.columns(3)

k1.metric(
    "Customers in View",
    f"{len(filtered_df):,}",
)

k2.metric(
    "Customer Segments",
    filtered_df["Segment"].nunique(),
)

if income_columns and not filtered_df.empty:
    k3.metric(
        "Average Income",
        f"{filtered_df[income_columns[0]].mean():,.2f}",
    )
else:
    k3.metric(
        "Numeric Features",
        len(numeric_df.columns),
    )

st.divider()

# =====================================================
# DASHBOARD TABS
# =====================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Segment Analysis",
    "Feature Explorer",
    "Correlation Analysis",
    "Model Evaluation",
    "Predict New Customers",
])

# =====================================================
# TAB 1 — SEGMENT ANALYSIS
# =====================================================
with tab1:
    st.subheader("Customer Segment Analysis")

    chosen_model = st.selectbox(
        "Choose Clustering Model",
        list(labels_by_model.keys()),
        key="segment_model",
    )

    result_df = df.copy()
    result_df["Segment"] = [
        "Noise" if label == -1 else f"Cluster {label}"
        for label in labels_by_model[chosen_model]
    ]

    visible_df = result_df.loc[mask]

    if visible_df.empty:
        st.warning("No customers match the selected filters.")
    else:
        counts = (
            visible_df["Segment"]
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )

        left, right = st.columns(2)

        with left:
            st.plotly_chart(
                px.bar(
                    counts,
                    x="Segment",
                    y="Customers",
                    color="Segment",
                    title="Customer Distribution",
                    template="plotly_dark",
                ),
                use_container_width=True,
            )

        with right:
            st.plotly_chart(
                px.pie(
                    counts,
                    names="Segment",
                    values="Customers",
                    hole=0.4,
                    title="Customer Segment Share",
                    template="plotly_dark",
                ),
                use_container_width=True,
            )

        st.subheader("Average Numeric Values by Segment")

        summary = numeric_df.copy()
        summary["Segment"] = result_df["Segment"].values

        st.dataframe(
            summary.loc[mask]
            .groupby("Segment")
            .mean(numeric_only=True)
            .round(2),
            use_container_width=True,
        )

        st.subheader("Customer Records")
        st.dataframe(visible_df.head(500), use_container_width=True)

        st.download_button(
            "Download Segmented Customers CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="segmented_customers.csv",
            mime="text/csv",
        )

# =====================================================
# TAB 2 — FEATURE EXPLORER
# =====================================================
with tab2:
    st.subheader("Feature Explorer")

    features = list(numeric_df.columns)

    x_feature = st.selectbox(
        "X-axis Feature",
        features,
        key="feature_x",
    )

    y_feature = st.selectbox(
        "Y-axis Feature",
        [col for col in features if col != x_feature],
        key="feature_y",
    )

    color_model = st.selectbox(
        "Color by Model",
        list(labels_by_model.keys()),
        key="feature_model",
    )

    plot_df = df.copy()
    plot_df["Cluster"] = [
        "Noise" if label == -1 else f"Cluster {label}"
        for label in labels_by_model[color_model]
    ]
    plot_df = plot_df.loc[mask]

    st.plotly_chart(
        px.scatter(
            plot_df,
            x=x_feature,
            y=y_feature,
            color="Cluster",
            title=f"{y_feature} vs {x_feature}",
            template="plotly_dark",
        ),
        use_container_width=True,
    )

    distribution_feature = st.selectbox(
        "Feature Distribution",
        features,
        key="distribution_feature",
    )

    st.plotly_chart(
        px.histogram(
            numeric_df.loc[mask],
            x=distribution_feature,
            title=f"Distribution of {distribution_feature}",
            template="plotly_dark",
        ),
        use_container_width=True,
    )

# =====================================================
# TAB 3 — CORRELATION ANALYSIS
# =====================================================
with tab3:
    st.subheader("Correlation Analysis")

    correlation_data = numeric_df.loc[mask]

    if len(correlation_data) < 2:
        correlation_data = numeric_df

    st.plotly_chart(
        px.imshow(
            correlation_data.corr(),
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Feature Correlation Heatmap",
            template="plotly_dark",
        ),
        use_container_width=True,
    )

    st.caption(
        "Correlation values range from -1 to +1. "
        "Correlation alone does not prove causation."
    )

# =====================================================
# TAB 4 — MODEL EVALUATION
# =====================================================
with tab4:
    st.subheader("Model Evaluation Comparison")

    evaluation_df = pd.DataFrame(evaluation_rows)

    st.dataframe(
        evaluation_df,
        use_container_width=True,
        hide_index=True,
    )

    valid_scores = evaluation_df.dropna(
        subset=["Silhouette Score"]
    )

    if not valid_scores.empty:
        st.plotly_chart(
            px.bar(
                valid_scores,
                x="Model",
                y="Silhouette Score",
                color="Model",
                title="Silhouette Score Comparison",
                template="plotly_dark",
            ),
            use_container_width=True,
        )

    st.markdown("""
    **Metric interpretation**

    - Silhouette Score: higher is generally better.
    - Davies-Bouldin Index: lower is generally better.
    - Calinski-Harabasz Score: higher is generally better.
    - DBSCAN marks noise records with the label `-1`.
    """)

    st.download_button(
        "Download Model Evaluation CSV",
        evaluation_df.to_csv(index=False).encode("utf-8"),
        file_name="model_evaluation.csv",
        mime="text/csv",
    )

# =====================================================
# TAB 5 — PREDICT NEW CUSTOMERS
# =====================================================
with tab5:
    st.subheader("Predict New Customers")

    st.write(
        "Enter values for the numeric features. "
        "The app will assign the new customer to a segment."
    )

    prediction_model = st.selectbox(
        "Choose Model",
        list(labels_by_model.keys()),
        key="prediction_model",
    )

    st.caption(
        "K-Means uses native prediction. For DBSCAN, Birch and "
        "Spectral Clustering, assignment uses a nearest-cluster-centre "
        "approximation."
    )

    with st.form("new_customer_form"):
        form_columns = st.columns(3)
        new_values = {}

        for i, feature in enumerate(numeric_df.columns):
            series = numeric_df[feature].dropna()
            default_value = (
                float(series.median()) if len(series) else 0.0
            )

            with form_columns[i % 3]:
                new_values[feature] = st.number_input(
                    str(feature),
                    value=default_value,
                    key=f"new_customer_{feature}",
                )

        submitted = st.form_submit_button(
            "Assign Customer Segment",
            type="primary",
        )

    if submitted:
        new_df = pd.DataFrame(
            [new_values],
            columns=numeric_df.columns,
        )
        new_X = scaler.transform(new_df)

        if prediction_model == "K-Means":
            prediction = str(
                int(models["K-Means"].predict(new_X)[0])
            )
            method = "K-Means native prediction"
        else:
            prediction = nearest_cluster(
                X,
                labels_by_model[prediction_model],
                new_X,
            )
            method = "Nearest cluster-centre approximation"

        st.success(
            f"Assigned Customer Segment: Cluster {prediction}"
        )
        st.caption(f"Assignment method: {method}")

        prediction_df = new_df.copy()
        prediction_df["Predicted Segment"] = f"Cluster {prediction}"

        st.dataframe(prediction_df, use_container_width=True)

        st.download_button(
            "Download Prediction CSV",
            prediction_df.to_csv(index=False).encode("utf-8"),
            file_name="new_customer_prediction.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Customer Segmentation Dashboard | "
    "K-Means • DBSCAN • Birch • Spectral Clustering"
)