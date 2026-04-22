import os
import re
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from supertree import SuperTree


COLOUR_SEQUENCE = ["#0073C0", "#AD00BD"]
CLASS_NAMES = ["Legitimate", "Fraudulent"]
TARGET_COLUMN = "fraud"
FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "merchant_risk",
    "device_trusted",
    "international",
    "card_present",
    "transactions_last_24h",
    "account_age_days",
]
NUMERIC_COLUMNS = [
    "amount",
    "hour_of_day",
    "transactions_last_24h",
    "account_age_days",
]
CATEGORICAL_COLUMNS = [
    "merchant_risk",
    "device_trusted",
    "international",
    "card_present",
]
DATASET_FILES = {
    "Dataset A - Poor Quality": "fraud_dataset_A_poor_quality.csv",
    "Dataset B - Imbalanced": "fraud_dataset_B_imbalanced.csv",
    "Dataset C - Good Quality": "fraud_dataset_C_good_quality.csv",
}
TEST_FILE = "fraud_test_balanced.csv"
DATASET_DESCRIPTIONS = {
    "Dataset A - Poor Quality": "This dataset contains duplicates, missing values, inconsistent category labels and corrupted values. The tree is trained on the raw dataset with only minimal technical preprocessing so those issues still affect the model.",
    "Dataset B - Imbalanced": "This dataset is comparatively clean, but fraudulent transactions are very rare. It demonstrates how a decision tree can appear to perform well while failing to identify the minority class.",
    "Dataset C - Good Quality": "This dataset is clean, more balanced and has a stronger fraud signal. It is intended to show what a more useful and interpretable tree looks like when the data is ready for modelling.",
}

MONITORING_SCENARIOS = {
    "Scenario 1": "Scenario A - Stable",
    "Scenario 2": "Scenario B - Feature Drift",
    "Scenario 3": "Scenario C - Concept Drift",
}
DATASET_LABELS = {
    "Dataset 1": "Dataset A - Poor Quality",
    "Dataset 2": "Dataset B - Imbalanced",
    "Dataset 3": "Dataset C - Good Quality",
}

@st.cache_data
def load_csv_from_known_locations(filename: str) -> pd.DataFrame:
    """Load a CSV from one of the known local deployment paths."""
    candidate_dirs = [Path("../Supervised-Data"), Path("."), Path(__file__).resolve().parent / "../Supervised-Data", Path(__file__).resolve().parent]
    for directory in candidate_dirs:
        path = directory / filename
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Could not find '{filename}'. Place it beside the app file or inside a ../Supervised-Data/ folder."
    )


@st.cache_resource
def get_one_hot_encoder() -> OneHotEncoder:
    """Create a version-compatible one-hot encoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_data
def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load and standardise a named training dataset."""
    df = load_csv_from_known_locations(DATASET_FILES[dataset_name]).copy()
    return standardise_dataframe(df)


@st.cache_data
def load_test_dataset() -> pd.DataFrame:
    """Load and standardise the shared test dataset."""
    df = load_csv_from_known_locations(TEST_FILE).copy()
    return standardise_dataframe(df)


def standardise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the required schema and tidy column names."""
    df.columns = [str(c).strip() for c in df.columns]
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
    return df


def get_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add display-friendly target labels for plotting."""
    display_df = df.copy()
    display_df["target_name"] = display_df[TARGET_COLUMN].map({0: CLASS_NAMES[0], 1: CLASS_NAMES[1]})
    return display_df


def build_pipeline(max_depth: int) -> Pipeline:
    """Build the preprocessing and decision-tree pipeline."""
    # numeric_pipeline = Pipeline([
    #     ("imputer", SimpleImputer(strategy="median")),
    # ])
    categorical_pipeline = Pipeline([
        #("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", get_one_hot_encoder()),
    ])
    preprocessor = ColumnTransformer([
        #("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ],remainder="passthrough")
    model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def fit_model(df: pd.DataFrame, max_depth: int) -> tuple[Pipeline, DecisionTreeClassifier, np.ndarray, np.ndarray, list[str]]:
    """Fit a model and return the fitted components used across the app."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    pipeline = build_pipeline(max_depth)
    pipeline.fit(X, y)

    transformed_X = pipeline.named_steps["preprocessor"].transform(X)
    feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    feature_names = [f.split("__")[-1] for f in feature_names]
    clf = pipeline.named_steps["model"]
    return pipeline, clf, transformed_X, y, feature_names


def ensure_trained_models_store() -> None:
    """Ensure the per-dataset trained model store exists in session state."""
    if "trained_models" not in st.session_state:
        st.session_state["trained_models"] = {}


def save_trained_model_bundle(
    dataset_name: str,
    dataset_label: str,
    pipeline: Pipeline,
    clf: DecisionTreeClassifier,
    X_train_transformed: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    viz_html: str,
    test_predictions: np.ndarray,
    model_test_acc: float,
) -> None:
    """Store a trained model bundle for the selected dataset in session state."""
    ensure_trained_models_store()
    st.session_state["trained_models"][dataset_name] = {
        "dataset_name": dataset_name,
        "dataset_label": dataset_label,
        "pipeline": pipeline,
        "clf": clf,
        "X_train_transformed": X_train_transformed,
        "y_train": y_train,
        "feature_names": feature_names,
        "viz_html": viz_html,
        "test_predictions": test_predictions,
        "model_test_acc": model_test_acc,
    }


def get_trained_model_bundle(dataset_name: str) -> dict[str, Any] | None:
    """Return the stored model bundle for a dataset if available."""
    ensure_trained_models_store()
    return st.session_state["trained_models"].get(dataset_name)


def has_trained_model(dataset_name: str) -> bool:
    ensure_trained_models_store()
    return dataset_name in st.session_state["trained_models"]


def clear_trained_model(dataset_name: str | None = None) -> None:
    """Clear one stored model bundle, or all if no dataset is provided."""
    ensure_trained_models_store()
    if dataset_name is None:
        st.session_state["trained_models"] = {}
    else:
        st.session_state["trained_models"].pop(dataset_name, None)

def clear_all_trained_models() -> None:
    """Clear all stored trained model bundles."""
    st.session_state["trained_models"] = {}

# def clear_trained_model():
#     keys_to_clear = [
#         "trained_pipeline",
#         "trained_model",
#         "feature_names",
#         "X_train_transformed",
#         "y_train",
#         "dataset_name",
#         "viz_html",
#         "model_trained",
#     ]
#     for key in keys_to_clear:
#         if key in st.session_state:
#             del st.session_state[key]

def inject_supertree_class_colors(html_content: str) -> str:
    """
    Replace SuperTree's internal colour palette with one derived from the app's
    class colours. SuperTree uses the JS array `M=[...]` and slices it into
    `ot` for classification colouring, so patching M is the reliable fix.
    """
    # Build a 20-colour palette where the first two colours are your class colours
    # and the rest alternate, so every palette slice still stays on-brand.
    custom_palette = [COLOUR_SEQUENCE[i % len(COLOUR_SEQUENCE)] for i in range(20)]
    palette_js = json.dumps(custom_palette)

    # Replace the SuperTree palette definition.
    # The generated HTML contains: const M=[...]
    html_content, n = re.subn(
        r'const M=\[(.*?)\]',
        f'const M={palette_js}',
        html_content,
        count=1,
        flags=re.S,
    )

    # Optional safety check during development
    #print(f"Patched SuperTree palette definitions: {n}")

    return html_content


def get_current_page() -> str:
    """Return the currently selected page from the sidebar."""
    return st.sidebar.selectbox("Page", ["Data/Model Explorer", "Make Predictions", "Model Evaluation", "Model Comparison","Continuous Monitoring"], index=0)

def generate_vis(
    clf: DecisionTreeClassifier,
    X_transformed: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    sample: np.ndarray | None = None,
) -> str:
    """Generate SuperTree HTML for a trained classifier."""
    super_tree = SuperTree(clf, X_transformed, y, feature_names, class_names)
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".html", delete=False, encoding="utf-8") as tmp_file:
        html_filename = tmp_file.name
    try:
        super_tree.save_html(html_filename)
        with open(html_filename, "r", encoding="utf-8") as f:
            html_content = f.read()
    finally:
        if os.path.exists(html_filename):
            os.remove(html_filename)

    html_content = inject_supertree_class_colors(html_content)
    if sample is not None:
        html_content = inject_decision_path_css(html_content, clf, sample)
    return html_content


def inject_decision_path_css(html_content: str, clf: DecisionTreeClassifier, sample: np.ndarray) -> str:
    """Inject CSS to highlight a sample path in the SuperTree HTML."""
    sample = np.array(sample).reshape(1, -1)
    decision_path = clf.decision_path(sample)
    node_ids = decision_path.nonzero()[1]
    css_rules = "\n".join([f"#node{node} {{ border: 2px solid orange !important; }}" for node in node_ids])
    return f"<style>{css_rules}</style>" + html_content


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex colour string to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convert an RGB tuple to a hex colour string."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def blend_colors(color1: str, color2: str, ratio: float) -> str:
    """Blend two colours by a given ratio."""
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    r = int(r1 + (r2 - r1) * ratio)
    g = int(g1 + (g2 - g1) * ratio)
    b = int(b1 + (b2 - b1) * ratio)
    return rgb_to_hex((r, g, b))


def get_path_in_order(clf: DecisionTreeClassifier, sample: np.ndarray) -> list[int]:
    """Return the node path followed by a sample through the tree."""
    tree = clf.tree_
    node = 0
    path = []
    while tree.children_left[node] != -1:
        path.append(node)
        feature = tree.feature[node]
        threshold = tree.threshold[node]
        if sample[0, feature] <= threshold:
            node = tree.children_left[node]
        else:
            node = tree.children_right[node]
    path.append(node)
    return path


def get_decision_path_edges(clf: DecisionTreeClassifier, sample: np.ndarray) -> list[tuple[int, int]]:
    """Return the ordered edges traversed by a sample through the tree."""
    path = get_path_in_order(clf, sample)
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def highlight_dot(
    clf: DecisionTreeClassifier,
    sample: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
) -> str:
    """Return Graphviz DOT with the sample decision path highlighted."""
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
    )

    path_nodes = set(get_path_in_order(clf, sample))
    path_edges = set(get_decision_path_edges(clf, sample))

    node_pattern = re.compile(r"^(\s*(\d+)\s*\[)(.*)(\]\s*;?\s*)$")
    edge_pattern = re.compile(r"^(\s*)(\d+)\s*->\s*(\d+)(\s*\[([^\]]*)\])?\s*;?\s*$")

    new_lines = []
    for line in dot_data.splitlines():
        node_match = node_pattern.match(line)
        if node_match:
            node_id = int(node_match.group(2))
            attr_str = node_match.group(3)
            suffix = node_match.group(4)
            if node_id in path_nodes:
                new_attr = re.sub(r'(,\s*)?\b(penwidth|color)\b\s*=\s*("[^"]*"|[^,\]]+)', "", attr_str)
                new_attr = new_attr.strip()
                new_attr = re.sub(r"\s*,\s*,+", ",", new_attr)
                new_attr = re.sub(r"^\s*,", "", new_attr)
                new_attr = re.sub(r",\s*$", "", new_attr)
                custom_node_style = 'penwidth=3, color="red"'
                new_attr = f"{new_attr}, {custom_node_style}" if new_attr else custom_node_style
                line = f"{node_match.group(1)}{new_attr}{suffix}"
            counts = clf.tree_.value[node_id][0]
            total = counts.sum()
            if total > 0 and np.sum(counts == counts.max()) == 1:
                majority = counts.max()
                second_largest = np.partition(counts, -2)[-2]
                ratio = (majority - second_largest) / majority if majority else 0
            else:
                ratio = 0
            if ratio == 0:
                new_fill = "#FFFFFF"
            else:
                majority_class = int(np.argmax(counts))
                base_color = COLOUR_SEQUENCE[majority_class]
                new_fill = blend_colors("#FFFFFF", base_color, ratio)
            if 'fillcolor=' in line:
                line = re.sub(r'fillcolor="[^"]*"', f'fillcolor="{new_fill}"', line)
            else:
                line = line.rstrip("]") + f', fillcolor="{new_fill}"]'
            new_lines.append(line)
            continue

        edge_match = edge_pattern.match(line)
        if edge_match:
            src = int(edge_match.group(2))
            dst = int(edge_match.group(3))
            if (src, dst) in path_edges:
                attr_content = edge_match.group(5) if edge_match.group(5) is not None else ""
                attr_content = re.sub(r'(,\s*)?(color|penwidth)\s*=\s*("[^"]*"|[^,\]]+)', "", attr_content)
                attr_content = attr_content.strip()
                attr_content = re.sub(r"^\s*,", "", attr_content)
                attr_content = re.sub(r",\s*$", "", attr_content)
                custom_edge_style = 'color="red", penwidth=3'
                new_attr = f"{attr_content}, {custom_edge_style}" if attr_content else custom_edge_style
                line = f"{edge_match.group(1)}{edge_match.group(2)} -> {edge_match.group(3)} [{new_attr}];"
            new_lines.append(line)
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


def compute_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """Compute the core classification metrics used across the app."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Fraud)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Fraud)": recall_score(y_true, y_pred, zero_division=0),
        "F1-score (Fraud)": f1_score(y_true, y_pred, zero_division=0),
    }


def plot_metrics_bar(metrics: dict[str, float], title: str = "Metric Scores") -> None:
    """Plot a compact metrics bar chart."""
    metric_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Score": list(metrics.values()),
    })
    fig = px.bar(
        metric_df,
        x="Metric",
        y="Score",
        color="Score",
        text="Score",
        color_continuous_scale="Blues",
        range_color=[0, 1],
        title=title if title else None,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis=dict(range=[0, 1], tickformat=".1f"),
        xaxis_title="",
        yaxis_title="Score",
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_class_balance(df: pd.DataFrame) -> None:
    """Plot the target class balance for a dataset."""
    counts = (
        df[TARGET_COLUMN]
        .map({0: CLASS_NAMES[0], 1: CLASS_NAMES[1]})
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="Count")
    )
    fig = px.bar(
        counts,
        x="Class",
        y="Count",
        color="Class",
        color_discrete_sequence=COLOUR_SEQUENCE[:2],
        text="Count",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# def show_scatter_matrix(df: pd.DataFrame):
#     plot_df = get_display_dataframe(df)
#     fig = px.scatter_matrix(
#         plot_df,
#         dimensions=NUMERIC_COLUMNS,
#         color="target_name",
#         hover_data=FEATURE_COLUMNS,
#         height=700,
#         color_discrete_sequence=COLOUR_SEQUENCE[:2],
#     )
#     fig.update_layout(margin=dict(l=50, r=50, t=50, b=50), autosize=True)
#     for axis in fig.layout:
#         if axis.startswith("xaxis"):
#             fig.layout[axis].tickangle = -45
#     st.plotly_chart(fig, use_container_width=True)


# def show_correlation_heatmap(df: pd.DataFrame):
#     corr_df = df[NUMERIC_COLUMNS + [TARGET_COLUMN]].copy()
#     corr = corr_df.corr(numeric_only=True)
#     fig = px.imshow(
#         corr,
#         text_auto=True,
#         aspect="auto",
#         color_continuous_scale="RdBu",
#         zmin=-1,
#         zmax=1,
#     )
#     st.plotly_chart(fig, use_container_width=True)


def show_feature_distributions(df: pd.DataFrame) -> None:
    """Render dataset visualisations and summary statistics."""
    plot_df = get_display_dataframe(df)

    tab1, tab2 = st.tabs(["Data Visualisations", "Dataset Summary"])

    with tab1:

        feature_type = st.radio(
            "Feature type",
            ("Numeric Features", "Categorical Features", "Target Feature"),
            horizontal=True,
        )
        typ, brk = st.columns(2)
        with typ:
            if feature_type == "Numeric Features":
                selected_features = st.multiselect(
                    "Choose numeric features to display",
                    NUMERIC_COLUMNS,
                    default=NUMERIC_COLUMNS[:2],
                )
            elif feature_type=="Categorical Features":
                selected_features = st.multiselect(
                    "Choose categorical features to display",
                    CATEGORICAL_COLUMNS,
                    default=CATEGORICAL_COLUMNS[:2],
                )

            
        with brk:
            if feature_type=="Target Feature":
                st.markdown(" ")
            else:
                st.markdown("By Target?")
                breakdown_by_target = st.toggle(
                    "Break down plots by target variable",
                    value=True,
                    help="When enabled, each feature is shown separately for legitimate and fraudulent transactions.",
                )


        if feature_type == "Numeric Features":
            if not selected_features:
                st.info("Select at least one numeric feature to display.")
            else:
                for feature in selected_features:
                    plot_data = plot_df[[feature, "target_name"]].copy()
                    plot_data = plot_data.dropna(subset=[feature])

                    if breakdown_by_target:
                        st.markdown(f"**{feature.replace('_', ' ').title()}**")
                        col1, col2 = st.columns(2)

                        for idx, class_name in enumerate(CLASS_NAMES):
                            subset = plot_data[plot_data["target_name"] == class_name]
                            fig = px.histogram(
                                subset,
                                x=feature,
                                nbins=30,
                                color_discrete_sequence=[COLOUR_SEQUENCE[idx]],
                            )
                            fig.update_layout(
                                margin=dict(l=20, r=20, t=40, b=20),
                                title=f"{feature.replace('_', ' ').title()} — {class_name}",
                                showlegend=False,
                            )
                            if idx == 0:
                                col1.plotly_chart(fig, use_container_width=True)
                            else:
                                col2.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.histogram(
                            plot_data,
                            x=feature,
                            nbins=30,
                            color_discrete_sequence=[COLOUR_SEQUENCE[0]],
                        )
                        fig.update_layout(
                            margin=dict(l=20, r=20, t=40, b=20),
                            title=feature.replace("_", " ").title(),
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

        elif feature_type=="Categorical Features":
            if not selected_features:
                st.info("Select at least one categorical feature to display.")
            else:
                for feature in selected_features:
                    plot_data = plot_df[[feature, "target_name"]].copy()
                    plot_data[feature] = plot_data[feature].fillna("Missing").astype(str)

                    if breakdown_by_target:
                        st.markdown(f"**{feature.replace('_', ' ').title()}**")
                        col1, col2 = st.columns(2)

                        for idx, class_name in enumerate(CLASS_NAMES):
                            subset = plot_data[plot_data["target_name"] == class_name]
                            fig = px.histogram(
                                subset,
                                x=feature,
                                color_discrete_sequence=[COLOUR_SEQUENCE[idx]],
                            )
                            fig.update_layout(
                                margin=dict(l=20, r=20, t=40, b=20),
                                title=f"{feature.replace('_', ' ').title()} — {class_name}",
                                showlegend=False,
                            )
                            fig.update_xaxes(categoryorder="total descending")
                            if idx == 0:
                                col1.plotly_chart(fig, use_container_width=True)
                            else:
                                col2.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.histogram(
                            plot_data,
                            x=feature,
                            color_discrete_sequence=[COLOUR_SEQUENCE[0]],
                        )
                        fig.update_layout(
                            margin=dict(l=20, r=20, t=40, b=20),
                            title=feature.replace("_", " ").title(),
                            showlegend=False,
                        )
                        fig.update_xaxes(categoryorder="total descending")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            plot_class_balance(df)

    with tab2:
        st.markdown("### Dataset Summary")

        summary_by_target = st.toggle(
            "Break down descriptive statistics by target variable",
            value=False,
            help="When enabled, separate summary tables are shown for legitimate and fraudulent transactions.",
            key="summary_by_target_toggle",
        )

        if not summary_by_target:
            numeric_summary = (
                df[NUMERIC_COLUMNS]
                .describe()
                .T
                .rename(columns={
                    "count": "Count",
                    "mean": "Mean",
                    "std": "Std Dev",
                    "min": "Min",
                    "25%": "Q1",
                    "50%": "Median",
                    "75%": "Q3",
                    "max": "Max",
                })
                .reset_index()
                .rename(columns={"index": "Feature"})
            )

            st.markdown("### 📈 Numerical Features")
            st.dataframe(
                numeric_summary.style.format({
                    "Count": "{:.0f}",
                    "Mean": "{:.2f}",
                    "Std Dev": "{:.2f}",
                    "Min": "{:.2f}",
                    "Q1": "{:.2f}",
                    "Median": "{:.2f}",
                    "Q3": "{:.2f}",
                    "Max": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True
            )
            st.caption("Q1 = 25th percentile, Median = 50th percentile, Q3 = 75th percentile")

            categorical_df = df[CATEGORICAL_COLUMNS]#.fillna("Missing").astype(str)
            categorical_summary = (
                categorical_df
                .describe(include="all")
                .T
                .rename(columns={
                    "count": "Count",
                    "unique": "Unique Values",
                    "top": "Most Common",
                    "freq": "Frequency",
                })
                .reset_index()
                .rename(columns={"index": "Feature"})
            )

            st.markdown("### 📊 Categorical Features")
            st.dataframe(categorical_summary, use_container_width=True,
                hide_index=True)

        else:
            summary_df = get_display_dataframe(df)

            col1, col2 = st.columns(2)

            for idx, (col, class_name) in enumerate(zip([col1, col2], CLASS_NAMES)):
                with col:
                    class_df = summary_df[summary_df["target_name"] == class_name].copy()

                    st.markdown(
                        f"## <span style='color:{COLOUR_SEQUENCE[idx]}'>{class_name}</span>",
                        unsafe_allow_html=True,
                    )

                    # --- Numerical ---
                    numeric_summary = (
                        class_df[NUMERIC_COLUMNS]
                        .describe()
                        .T
                        .rename(columns={
                            "count": "Count",
                            "mean": "Mean",
                            "std": "Std Dev",
                            "min": "Min",
                            "25%": "Q1",
                            "50%": "Median",
                            "75%": "Q3",
                            "max": "Max",
                        })
                        .reset_index()
                        .rename(columns={"index": "Feature"})
                    )

                    if col==col1:
                        st.markdown("### 📈 Numerical Features")
                    else:
                        st.markdown("###  ")
                    st.dataframe(
                        numeric_summary.style.format({
                            "Count": "{:.0f}",
                            "Mean": "{:.2f}",
                            "Std Dev": "{:.2f}",
                            "Min": "{:.2f}",
                            "Q1": "{:.2f}",
                            "Median": "{:.2f}",
                            "Q3": "{:.2f}",
                            "Max": "{:.2f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # --- Categorical ---
                    categorical_df = class_df[CATEGORICAL_COLUMNS]#.fillna("Missing").astype(str)
                    categorical_summary = (
                        categorical_df
                        .describe(include="all")
                        .T
                        .rename(columns={
                            "count": "Count",
                            "unique": "Unique Values",
                            "top": "Most Common",
                            "freq": "Frequency",
                        })
                        .reset_index()
                        .rename(columns={"index": "Feature"})
                    )

                    if col==col1:
                        st.markdown("### 📊 Categorical Features")
                    else:
                        st.markdown("###  ")
                    st.dataframe(categorical_summary, use_container_width=True,hide_index=True)

            st.caption("Q1 = 25th percentile, Median = 50th percentile, Q3 = 75th percentile")


def show_data_visualisation(df: pd.DataFrame) -> None:
    """Render the dataset exploration view."""

    show_feature_distributions(df)


def dataset_health_summary(df: pd.DataFrame) -> None:
    """Show headline dataset health indicators."""


    duplicate_rows = int(df.duplicated().sum())
    missing_values = int(df[FEATURE_COLUMNS].isna().sum().sum())
    invalid_amounts = int((pd.to_numeric(df["amount"], errors="coerce") < 0).fillna(False).sum())
    fraud_rate = float(df[TARGET_COLUMN].mean())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Fraud Rate", f"{fraud_rate:.1%}")
    col3.metric("Missing Values", f"{missing_values:,}")
    col4.metric("Duplicate Rows", f"{duplicate_rows:,}")
    # if invalid_amounts > 0:
    #     st.caption(f"This dataset contains {invalid_amounts} transaction amounts below zero.")


def display_metrics(metrics: dict[str, float], title: str) -> None:
    """Display metrics as headline summary values."""
    st.markdown(f"**{title}**")
    cols = st.columns(len(metrics))
    for idx, (label, value) in enumerate(metrics.items()):
        cols[idx].metric(label, f"{value:.3f}")


def plot_confusion(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, title: str) -> None:
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        zmin=0,
        zmax=max(int(cm_df.to_numpy().max()), 1),
        aspect="auto",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title=title if title else None,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def compare_all_datasets(max_depth: int, test_df: pd.DataFrame) -> pd.DataFrame:
    """Train and evaluate each dataset for anonymous side-by-side comparison."""
    rows = []
    reverse_labels = {value: key for key, value in DATASET_LABELS.items()}
    for dataset_name in DATASET_FILES:
        train_df = load_dataset(dataset_name)
        pipeline, clf, X_train_transformed, y_train, feature_names = fit_model(train_df, max_depth)
        y_train_pred = pipeline.predict(train_df[FEATURE_COLUMNS])
        y_test_pred = pipeline.predict(test_df[FEATURE_COLUMNS])
        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(test_df[TARGET_COLUMN].astype(int), y_test_pred)
        rows.append({
            "Dataset": reverse_labels.get(dataset_name, dataset_name),
            "Train Accuracy": train_metrics["Accuracy"],
            "Test Accuracy": test_metrics["Accuracy"],
            "Test Precision (Fraud)": test_metrics["Precision (Fraud)"],
            "Test Recall (Fraud)": test_metrics["Recall (Fraud)"],
            "Test F1-score (Fraud)": test_metrics["F1-score (Fraud)"],
        })
    return pd.DataFrame(rows)


def make_prediction_input(defaults: pd.Series) -> tuple[pd.DataFrame, bool]:
    """Render prediction inputs in the main page and return the values plus submit state."""
    merchant_risk_options = sorted(
        defaults.get("merchant_risk_options", ["High", "Low", "Medium"])
    )

    st.subheader("Enter Transaction Details")
    st.markdown("Adjust the transaction features below, then click **Make Prediction**.")

    with st.form("prediction_form", clear_on_submit=False):
        details_col, merchant_col, customer_col, activity_col = st.columns(4)

        with details_col:
            with st.container(border=True):
                st.markdown("**Transaction Details**")
                amount = st.number_input(
                    "Transaction Amount",
                    min_value=0.0,
                    value=float(defaults.get("amount", 120.0)),
                    step=1.0,
                )
                hour_of_day = st.number_input(
                    "Hour of Day",
                    min_value=0,
                    max_value=23,
                    value=int(defaults.get("hour_of_day", 12)),
                    step=1,
                )

        with merchant_col:
            with st.container(border=True):
                st.markdown("**Merchant & Channel**")
                merchant_risk = st.selectbox(
                    "Merchant Risk",
                    options=merchant_risk_options,
                    index=merchant_risk_options.index("Medium") if "Medium" in merchant_risk_options else 0,
                )
                card_present = st.selectbox(
                    "Card Present?",
                    options=["Yes", "No"],
                    index=0,
                )

        with customer_col:
            with st.container(border=True):
                st.markdown("**Customer Context**")
                device_trusted = st.selectbox(
                    "Trusted Device?",
                    options=["Yes", "No"],
                    index=0,
                )
                account_age_days = st.number_input(
                    "Account Age (days)",
                    min_value=0,
                    value=int(defaults.get("account_age_days", 365)),
                    step=1,
                )

        with activity_col:
            with st.container(border=True):
                st.markdown("**Activity Details**")
                international = st.selectbox(
                    "International Transaction?",
                    options=["No", "Yes"],
                    index=0,
                )
                transactions_last_24h = st.number_input(
                    "Transactions in Last 24 Hours",
                    min_value=0,
                    value=int(defaults.get("transactions_last_24h", 2)),
                    step=1,
                )

            submitted = st.form_submit_button(
                "Make Prediction",
                use_container_width=True,
            )

    input_df = pd.DataFrame([
        {
            "amount": amount,
            "hour_of_day": hour_of_day,
            "merchant_risk": merchant_risk,
            "device_trusted": device_trusted,
            "international": international,
            "card_present": card_present,
            "transactions_last_24h": transactions_last_24h,
            "account_age_days": account_age_days,
        }
    ])

    return input_df, submitted


@st.cache_resource
def train_good_baseline_model(max_depth: int = 3) -> dict[str, Any]:
    """Train and cache the fixed monitoring baseline model."""
    good_df = load_dataset("Dataset C - Good Quality")
    pipeline, clf, X_train_transformed, y_train, feature_names = fit_model(good_df, max_depth)
    return {
        "pipeline": pipeline,
        "clf": clf,
        "train_df": good_df,
        "X_train_transformed": X_train_transformed,
        "y_train": y_train,
        "feature_names": feature_names,
    }


def generate_monitoring_scenario_batch(
    scenario: str,
    period: int,
    batch_size: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic monitoring batch for the chosen scenario and period."""
    rng = np.random.default_rng(random_state + period)

    df = pd.DataFrame({
        "amount": rng.gamma(2.0, 120.0, batch_size),
        "hour_of_day": rng.integers(0, 24, batch_size),
        "merchant_risk": rng.choice(["Low", "Medium", "High"], batch_size, p=[0.3, 0.4, 0.3]),
        "device_trusted": rng.choice(["Yes", "No"], batch_size, p=[0.6, 0.4]),
        "international": rng.choice(["Yes", "No"], batch_size, p=[0.4, 0.6]),
        "card_present": rng.choice(["Yes", "No"], batch_size, p=[0.5, 0.5]),
        "transactions_last_24h": rng.poisson(4, batch_size),
        "account_age_days": rng.integers(10, 2000, batch_size),
    })

    # Scenario A - Stable
    if scenario == "Scenario A - Stable":
        pass

    # Scenario B - Feature Drift
    elif scenario == "Scenario B - Feature Drift":
        drift_strength = period / 12
        # Amounts gradually increase
        df["amount"] = df["amount"] * (1 + 0.5 * drift_strength)

        # Transactions gradually shift toward later hours
        shifted_hours = df["hour_of_day"] + rng.integers(0, int(6 * drift_strength) + 1, batch_size)
        df["hour_of_day"] = np.clip(shifted_hours, 0, 23)

        # Other mild distribution drift
        df["international"] = rng.choice(
            ["Yes", "No"],
            batch_size,
            p=[0.4 + 0.35 * drift_strength, 0.6 - 0.35 * drift_strength],
        )
        df["transactions_last_24h"] = df["transactions_last_24h"] + rng.poisson(2 * drift_strength, batch_size)

    # Scenario C - Concept Drift
    elif scenario == "Scenario C - Concept Drift":
        drift_strength = period / 12
        df["amount"] = df["amount"] * (1 + 0.1 * drift_strength)

    # Original target rule (same as good-quality signal)
    base_score = (
        (df["amount"] > 250).astype(int) * 2
        + (df["merchant_risk"] == "High").astype(int) * 3
        + (df["international"] == "Yes").astype(int) * 2
        + (df["device_trusted"] == "No").astype(int) * 2
        + (df["transactions_last_24h"] > 5).astype(int) * 1
        + (df["account_age_days"] < 200).astype(int) * 1
    )

    # Concept drift: later periods use a changed fraud rule
    if scenario == "Scenario C - Concept Drift" and period >= 4:
        score = (
            (df["amount"] > 320).astype(int) * 1
            + (df["merchant_risk"] == "High").astype(int) * 2
            + (df["international"] == "Yes").astype(int) * 1
            + (df["card_present"] == "No").astype(int) * 2
            + (df["hour_of_day"].isin([0, 1, 2, 3, 4])).astype(int) * 2
            + (df["transactions_last_24h"] > 7).astype(int) * 2
        )
    else:
        score = base_score

    prob = score / max(score.max(), 1)
    prob = prob * 0.9 + rng.uniform(0, 0.1, batch_size)
    df[TARGET_COLUMN] = (prob > 0.45).astype(int)

    return df


def build_monitoring_history(
    scenario: str,
    pipeline: Pipeline,
    periods: int = 8,
    batch_size: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    """Build monitoring metrics, prediction mix and scenario batches over time."""
    rows = []
    prediction_rows = []
    batches = {}

    for period in range(1, periods + 1):
        batch_df = generate_monitoring_scenario_batch(
            scenario=scenario,
            period=period,
            batch_size=batch_size,
            random_state=42,
        )

        y_true = batch_df[TARGET_COLUMN].astype(int)
        y_pred = pipeline.predict(batch_df[FEATURE_COLUMNS])
        metrics = compute_metrics(y_true, y_pred)

        pred_labels = pd.Series(y_pred).map({0: CLASS_NAMES[0], 1: CLASS_NAMES[1]})
        pred_counts = (
            pred_labels.value_counts(normalize=True)
            .reindex(CLASS_NAMES, fill_value=0)
            .reset_index()
        )
        pred_counts.columns = ["Predicted Class", "Proportion"]
        pred_counts["Period"] = period

        prediction_rows.append(pred_counts)

        rows.append({
            "Period": period,
            "Accuracy": metrics["Accuracy"],
            "Precision (Fraud)": metrics["Precision (Fraud)"],
            "Recall (Fraud)": metrics["Recall (Fraud)"],
            "F1-score (Fraud)": metrics["F1-score (Fraud)"],
        })

        batches[period] = batch_df

    history_df = pd.DataFrame(rows)
    prediction_dist_df = pd.concat(prediction_rows, ignore_index=True)

    return history_df, prediction_dist_df, batches


def plot_monitoring_metric_trends(history_df: pd.DataFrame) -> None:
    """Plot monitoring metrics across periods."""
    metric_df = history_df.melt(
        id_vars=["Period"],
        value_vars=["Accuracy", "Precision (Fraud)", "Recall (Fraud)", "F1-score (Fraud)"],
        var_name="Metric",
        value_name="Score",
    )

    fig = px.line(
        metric_df,
        x="Period",
        y="Score",
        color="Metric",
        markers=True,
        title="Performance Over Time",
    )
    fig.update_layout(yaxis=dict(range=[0, 1], tickformat=".1f"))
    st.plotly_chart(fig, use_container_width=True)


def plot_prediction_distribution_over_time(prediction_dist_df: pd.DataFrame) -> None:
    """Plot how the model prediction mix evolves over time."""
    fig = px.bar(
        prediction_dist_df,
        x="Period",
        y="Proportion",
        color="Predicted Class",
        barmode="stack",
        color_discrete_map={
            CLASS_NAMES[0]: COLOUR_SEQUENCE[0],
            CLASS_NAMES[1]: COLOUR_SEQUENCE[1],
        },
        title="Distribution of Model Predictions Over Time",
    )
    fig.update_layout(yaxis=dict(range=[0, 1], tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_drift_over_time(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature: str,
    period: int,
    normalise: bool,
) -> None:
    """Plot reference versus current feature distributions for drift inspection."""
    ref_plot = reference_df[[feature]].copy()
    ref_plot["Dataset"] = "Reference (Good Data)"
    cur_plot = current_df[[feature]].copy()
    cur_plot["Dataset"] = f"Period {period}"

    combined = pd.concat([ref_plot, cur_plot], ignore_index=True)
    if normalise:
        histnorm_a="probability density"
        histnorm_b="percent"
    else:
        histnorm_a=histnorm_b=None



    if feature in NUMERIC_COLUMNS:
        fig = px.histogram(
            combined,
            x=feature,
            color="Dataset",
            barmode="overlay",
            nbins=30,
            opacity=0.7,
            histnorm=histnorm_a,
            color_discrete_map={
                "Reference (Good Data)": COLOUR_SEQUENCE[0],
                f"Period {period}": COLOUR_SEQUENCE[1],
            },
            title=f"Feature Drift: {feature.replace('_', ' ').title()}",
        )
        
    else:
        combined[feature] = combined[feature].fillna("Missing").astype(str)
        fig = px.histogram(
            combined,
            x=feature,
            color="Dataset",
            barmode="group",
            histnorm=histnorm_b,
            color_discrete_map={
                "Reference (Good Data)": COLOUR_SEQUENCE[0],
                f"Period {period}": COLOUR_SEQUENCE[1],
            },
            title=f"Feature Drift: {feature.replace('_', ' ').title()}",
        )
    

    st.plotly_chart(fig, use_container_width=True)




def main() -> None:
    """Run the Streamlit application."""

    BASE_DIR = Path(__file__).resolve().parent
    ROOT_DIR = BASE_DIR.parent

    horizontal_logo = ROOT_DIR / "Logo.png"
    icon = ROOT_DIR / "Icon.png"

    st.set_page_config(
        page_title="Krisolis Data Quality, Evaluation and Monitoring Demonstration",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": """# Krisolis Data Quality, Evaluation and Monitoring Demonstration.
            Created by Eoghan Staunton @Krisolis
            Using scikit-learn, supertree, pandas and plotly""",
        },
    )
    if os.path.exists(horizontal_logo) and os.path.exists(icon):
        st.logo(horizontal_logo, size="large", link="http://www.krisolis.ie", icon_image=icon)

    st.title("Data Quality, Evaluation and Monitoring")
    st.markdown(
        "This app demonstrates how data quality, class balance, and evolving data conditions impact a **Decision Tree Classifier** in a fraud detection setting. Learners can explore datasets, train and compare models, make predictions, and investigate how model performance changes over time under different monitoring scenarios.",
    )

    st.sidebar.subheader("Choose Page")
    current_page = get_current_page()

    if current_page not in ["Model Comparison", "Continuous Monitoring"]:

        st.sidebar.subheader("Choose Dataset")
        dataset_label = st.sidebar.radio(
            "Training Dataset",
            options=list(DATASET_LABELS.keys()),
            help="Choose a dataset to explore before training a model",
        )

        dataset_name = DATASET_LABELS[dataset_label]
        
        st.session_state["dataset_name"] = dataset_name

    # st.sidebar.subheader("Set maximum tree depth")
    max_depth = 3
    # max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=4, value=3, step=1)

    try:
        if not st.session_state.get("dataset_name"):
            st.session_state["dataset_name"] = list(DATASET_FILES.keys())[0]
        selected_df = load_dataset(st.session_state["dataset_name"])
        test_df = load_test_dataset()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    if current_page == "Data/Model Explorer":
        st.header(
            f"Chosen Dataset: {dataset_label}",
            help="Explore the selected dataset before training the model.",
        )
        dataset_health_summary(selected_df)
        #st.info(DATASET_DESCRIPTIONS[dataset_name])

        

        st.sidebar.subheader("Train Model")
        
        if st.sidebar.button(
            "Train and Visualise Model",
            help="Train a decision tree using the selected dataset to see how data readiness/quality affects model performance",
        ):
            pipeline, clf, X_train_transformed, y_train, feature_names = fit_model(selected_df, max_depth)
            viz_html = generate_vis(clf, X_train_transformed, y_train, feature_names, CLASS_NAMES)
            test_predictions = pipeline.predict(test_df[FEATURE_COLUMNS])
            model_test_acc = accuracy_score(test_df[TARGET_COLUMN], test_predictions)

            save_trained_model_bundle(
                dataset_name=dataset_name,
                dataset_label=dataset_label,
                pipeline=pipeline,
                clf=clf,
                X_train_transformed=X_train_transformed,
                y_train=y_train,
                feature_names=feature_names,
                viz_html=viz_html,
                test_predictions=test_predictions,
                model_test_acc=model_test_acc,
            )

            st.toast(
                f"**Model Trained With** {dataset_label}.\n **Test Accuracy:** {model_test_acc:.3f}",
                icon="🤖",
            )

        model_bundle = get_trained_model_bundle(dataset_name)

        if model_bundle is not None:
            if st.sidebar.button("Reset to Data Exploration"):
                clear_trained_model(dataset_name)
                st.rerun()

        if model_bundle is None:
            show_data_visualisation(selected_df)
            st.sidebar.info(f"{dataset_label} has not been used to train a model yet.")
            return

        st.subheader(f"Trained Decision Tree on {dataset_label}")
        st.markdown(f"**Test Accuracy = {model_bundle['model_test_acc']:.3f}**")
        st.components.v1.html(model_bundle["viz_html"], height=600)

        if has_trained_model(dataset_name):
            st.sidebar.success(f"{dataset_label} already has a trained model in session.")
            


        

    elif current_page == "Model Comparison":
        st.header("Evaluation on Balanced Test Set")
        st.markdown("Compare how the same decision tree setup behaves when trained on dirty, imbalanced, and cleaner datasets.")
        comparison_df = compare_all_datasets(max_depth, test_df)
        metric_columns = [
            "Train Accuracy",
            "Test Accuracy",
            "Test Precision (Fraud)",
            "Test Recall (Fraud)",
            "Test F1-score (Fraud)",
        ]

        format_dict = {col: "{:.3f}" for col in metric_columns}

        styled_comparison_df = (
            comparison_df.style
            .format(format_dict)
            .set_properties(
                subset=["Dataset"],
                **{
                    "font-weight": "bold",
                    "text-align": "left",
                    "font-size": "15px",
                    "white-space": "nowrap",
                }
            )
            .set_properties(
                subset=metric_columns,
                **{
                    "text-align": "center",
                }
            )
            .background_gradient(
                cmap="Blues",   # matches your confusion matrix / metric bar styling
                subset=metric_columns,
                vmin=0,
                vmax=1,
            )
            # .highlight_max(
            #     subset=metric_columns,
            #     color="#dbe9f6",
            # )
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("padding", "6px 10px"),
                    ],
                },
            ])
        )

        st.dataframe(
            styled_comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        
      

        chart_df = comparison_df.melt(
            id_vars=["Dataset"],
            value_vars=[
                "Train Accuracy",
                "Test Accuracy",
                "Test Precision (Fraud)",
                "Test Recall (Fraud)",
                "Test F1-score (Fraud)",
            ],
            var_name="Metric",
            value_name="Score",
        )
        fig = px.bar(
            chart_df,
            x="Metric",
            y="Score",
            color="Dataset",
            color_discrete_map={
                "Dataset 1": "#F60B75",
                "Dataset 2": "#FFA500",
                "Dataset 3": "#00D3CF",
            },
            barmode="group",
            text="Score",
            title="Model Metrics by Dataset",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
        fig.update_layout(
            yaxis=dict(range=[0, 1], tickformat=".1f"),
            xaxis_title="",
            yaxis_title="Score",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        

        
        st.plotly_chart(fig, use_container_width=True)

        

    elif current_page ==  "Model Evaluation":
        st.header("Model Evaluation")
        model_bundle = get_trained_model_bundle(dataset_name)
        if model_bundle is None:
            st.info("Train your chosen model on the Data/Model Explorer page first, then come back here to examine its metrics when evaluated on a balanced test set.")
            return
        else:
            test_predictions = model_bundle["test_predictions"]
            test_metrics = compute_metrics(test_df[TARGET_COLUMN].astype(int), test_predictions)
            display_metrics(test_metrics, "Metrics")

            left_col, right_col = st.columns(2)
            with left_col:
                
                plot_metrics_bar(test_metrics, "Test Set Metric Comparison")
            with right_col:
                st.markdown("**Confusion Matrix**")
                plot_confusion(test_df[TARGET_COLUMN].astype(int), test_predictions, "")

    elif current_page == "Continuous Monitoring":
        st.header("Continuous Monitoring")
        st.markdown(
            "This page simulates how a model trained on good-quality data behaves over time in production."
        )
        st.markdown(
            "Your task is to inspect the monitoring outputs and decide what kind of situation you are seeing."
        )

        monitoring_model = train_good_baseline_model(max_depth=max_depth)
        baseline_pipeline = monitoring_model["pipeline"]
        baseline_reference_df = monitoring_model["train_df"]

        scenario_label = st.selectbox(
            "Choose Scenario",
            list(MONITORING_SCENARIOS.keys()),
            help="Each scenario represents a different production monitoring situation.",
        )

        scenario = MONITORING_SCENARIOS[scenario_label]

        history_df, prediction_dist_df, scenario_batches = build_monitoring_history(
            scenario=scenario,
            pipeline=baseline_pipeline,
            periods=12,
            batch_size=300,
        )

        st.subheader("1. Performance Evolution")
        plot_monitoring_metric_trends(history_df)

        st.subheader("2. Distribution of Predictions Over Time")
        plot_prediction_distribution_over_time(prediction_dist_df)

        st.subheader("3. Feature Drift Inspection")
        drift_col1, drift_col2, drift_col3 = st.columns(3)
        with drift_col1:
            feature_to_monitor = st.selectbox(
                "Choose Feature to Monitor",
                FEATURE_COLUMNS,
            )
        with drift_col2:
            selected_period = st.slider(
                "Choose Period",
                min_value=1,
                max_value=12,
                value=12,
                step=1,
            )
        with drift_col3:
            st.markdown("Normalise")
            normalise = st.checkbox("Show Densities", help="Make the comparison based on proportions (densities) rather than counts.")
        plot_feature_drift_over_time(
            reference_df=baseline_reference_df,
            current_df=scenario_batches[selected_period],
            feature=feature_to_monitor,
            period=selected_period,
            normalise=normalise
        )

        # st.markdown("### Group Task")
        # st.info(
        #     """
        #     Based on these monitoring outputs, discuss:
        #     - Does this look stable, like feature drift, or like concept drift?
        #     - What evidence supports your conclusion?
        #     - Would you investigate the data, the model, or both?
        #     """
        # )


    else:
        st.header("Prediction Explorer")
        st.markdown("Use a trained model to score a new transaction and inspect its decision path.")
        model_bundle = get_trained_model_bundle(dataset_name)
        if model_bundle is None:    
            st.info("Train your chosen model on the Data/Model Explorer page first, then come back here to make a prediction.")
            return

        current_defaults = selected_df[FEATURE_COLUMNS].copy()
        merchant_risk_values = sorted(current_defaults["merchant_risk"].dropna().astype(str).unique().tolist())
        default_row = current_defaults.iloc[0].copy()
        default_row["merchant_risk_options"] = merchant_risk_values or ["High", "Low", "Medium"]
        transaction_input, submitted = make_prediction_input(default_row)

        if submitted:
            pipeline = model_bundle["pipeline"]
            clf = model_bundle["clf"]
            transformed_sample = pipeline.named_steps["preprocessor"].transform(transaction_input)
            prediction = int(pipeline.predict(transaction_input)[0])
            probabilities = pipeline.predict_proba(transaction_input)[0]

            st.subheader(f"🔮 Prediction Results - Predicted Class: {CLASS_NAMES[prediction]}")

            prob_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Probability": probabilities,
            }).sort_values(by="Probability", ascending=False)

            fig = px.bar(
                prob_df,
                x="Probability",
                y="Class",
                orientation="h",
                color="Class",
                color_discrete_map={CLASS_NAMES[0]: COLOUR_SEQUENCE[0], CLASS_NAMES[1]: COLOUR_SEQUENCE[1]},
            )
            fig.update_layout(
                yaxis=dict(tickfont=dict(size=14)),
                xaxis=dict(tickformat=".2f", range=[0, 1]),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1),
            )

            highlighted_dot = highlight_dot(
                clf,
                np.asarray(transformed_sample),
                model_bundle["feature_names"],
                CLASS_NAMES,
            )

            
            
            st.write("**📊 Fraud Probabilities:**")
            st.plotly_chart(fig, use_container_width=True)
        
            st.write("**🛤️ Decision Path:**")
            st.graphviz_chart(highlighted_dot)


if __name__ == "__main__":


    if "trained_models" not in st.session_state:
        st.session_state["trained_models"] = {}
    main()
