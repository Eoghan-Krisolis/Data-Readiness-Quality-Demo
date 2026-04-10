import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from supertree import SuperTree  # SuperTree expects the model object and extracts what it needs
import re

# Define the desired palette for classes.
COLOUR_SEQUENCE = ["#5CB0F4", "#C21789"]

def train_model(df, max_depth):
    X = df.drop(columns=["fraud"])
    y = df["fraud"]
    clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    clf.fit(X, y)
    return clf, X, y, list(X.columns), list(y.name)

def inject_decision_path_css(html_content, clf, sample):
    """
    Compute the decision path for the test sample and inject CSS rules into the HTML
    to highlight nodes along the path. We assume that nodes in the HTML are identified by
    an id of the form "nodeX" where X is the node id.
    """
    # Ensure sample is 2D.
    sample = np.array(sample).reshape(1, -1)
    # Compute the decision path; nonzero returns (row_indices, col_indices).
    dp = clf.decision_path(sample)
    node_ids = dp.nonzero()[1]  # get the column indices (node ids)
    # Create CSS rules that highlight each node.
    # Adjust the CSS as needed; here we set an orange border.
    css_rules = "\n".join([f"#node{node} {{ border: 2px solid orange !important; }}" for node in node_ids])
    css_block = f"<style>{css_rules}</style>"
    # Prepend the CSS block to the HTML content.
    return css_block + html_content

def generate_vis(clf, X, y, feature_names, class_names, sample=None):
    # Create a SuperTree object.
    super_tree = SuperTree(clf, X, y, feature_names, class_names)
    html_filename = "supertree.html"
    # If a sample is provided, attempt to highlight the decision path.
    if sample is not None:
        # NOTE: The 'show_sample' parameter is premium, so we avoid it.
        # Instead we just generate the default visualisation.
        super_tree.save_html(html_filename)
    else:
        super_tree.save_html(html_filename)
    with open(html_filename, "r", encoding="utf-8") as f:
        html_content = f.read()
    # If a sample is provided, inject CSS to highlight the decision path.
    if sample is not None:
        html_content = inject_decision_path_css(html_content, clf, sample)
    return html_content

def show_scatter_matrix(df, feature_names):
    fig = px.scatter_matrix(
        df,
        dimensions=feature_names,
        color="fraud_name",
        hover_data=df.columns,
        height=700,
        color_discrete_sequence=COLOUR_SEQUENCE
    )
    fig.update_layout(
        margin=dict(l=50, r=50, t=50, b=50),
        autosize=True
    )
    # Rotate x-axis tick labels for all subplots.
    for axis in fig.layout:
        if axis.startswith("xaxis"):
            fig.layout[axis].tickangle = -45
    st.plotly_chart(fig, use_container_width=True)

def show_correlation_heatmap(df, feature_names):
    # Compute correlation matrix of the features.
    corr = df[feature_names+["fraud"]].corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_stacked_histograms(df, feature_names):
    
    # Melt the DataFrame so that we have one row per observation per feature.
    df_melt = df.melt(id_vars=["fraud", "fraud_name"], value_vars=feature_names,
                      var_name="Feature", value_name="Value")
    # Use Plotly Express to create a facet-wrapped histogram.
    fig = px.histogram(
        df_melt,
        x="Value",
        color="fraud_name",
        facet_col="Feature",
        facet_col_wrap=2,
        barmode="stack",
        nbins=40,
        color_discrete_sequence=COLOUR_SEQUENCE
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_data_visualisation(df, feature_names):
    st.subheader("Initial Data Visualisation", 
                 help="Explore different interactive visualisations of the dataset before training a model.")
    vis_type = st.radio(
        "Select Data Visualisation Type:",
        ("Histograms", "Scatter Matrix", "Correlation Heatmap"), 
        horizontal=True
    )
    if vis_type == "Scatter Matrix":
        show_scatter_matrix(df, feature_names)
    elif vis_type == "Correlation Heatmap":
        show_correlation_heatmap(df, feature_names)
    elif vis_type == "Histograms":
        show_stacked_histograms(df, feature_names)

def get_path_in_order(clf, sample):
    """
    Walks through the tree from the root to the leaf for the given sample
    and returns a list of node IDs in order.
    """
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

def get_decision_path_edges(clf, sample):
    """
    Returns a list of (parent, child) tuples for the edges along the decision path.
    """
    path = get_path_in_order(clf, sample)
    edges = []
    for i in range(len(path) - 1):
        edges.append((path[i], path[i+1]))
    return edges



def hex_to_rgb(hex_color):
    """Convert a hex color string (e.g. "#ff7f0e") to an (r, g, b) tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """Convert an (r, g, b) tuple to a hex color string."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def blend_colors(color1, color2, ratio):
    """
    Blend two hex colors. Ratio is a float between 0 and 1:
    0 returns color1, 1 returns color2.
    """
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    r = int(r1 + (r2 - r1) * ratio)
    g = int(g1 + (g2 - g1) * ratio)
    b = int(b1 + (b2 - b1) * ratio)
    return rgb_to_hex((r, g, b))

def highlight_dot(clf, sample, feature_names, class_names):
    """
    Generates DOT code via export_graphviz and then modifies it so that:
      - Nodes on the decision path get a red border (penwidth=3, color="red")
        while keeping their original fillcolor.
      - Edges along the decision path are highlighted in red with penwidth=3.
      - Additionally, each node’s fillcolor is overridden based on the node's
        class distribution: if there is a unique majority, we blend white and the class
        color based on how dominant that majority is; otherwise, the node is white.
    """
    from sklearn.tree import export_graphviz

    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    
    # Helper functions to get the decision path.
    def get_path_in_order(clf, sample):
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

    def get_decision_path_edges(clf, sample):
        path = get_path_in_order(clf, sample)
        return [(path[i], path[i+1]) for i in range(len(path) - 1)]
    
    path_nodes = set(get_path_in_order(clf, sample))
    path_edges = set(get_decision_path_edges(clf, sample))
    
    node_pattern = re.compile(r'^(\s*(\d+)\s*\[)(.*)(\]\s*;?\s*)$')
    edge_pattern = re.compile(r'^(\s*)(\d+)\s*->\s*(\d+)(\s*\[([^\]]*)\])?\s*;?\s*$')
    
    new_lines = []
    for line in dot_data.splitlines():
        # --- Process node definitions ---
        node_match = node_pattern.match(line)
        if node_match:
            node_id = int(node_match.group(2))
            attr_str = node_match.group(3)
            suffix = node_match.group(4)
            # If this node is on the decision path, update its border style.
            if node_id in path_nodes:
                new_attr = re.sub(r'(,\s*)?\b(penwidth|color)\b\s*=\s*("[^"]*"|[^,\]]+)', '', attr_str)
                new_attr = new_attr.strip()
                new_attr = re.sub(r'\s*,\s*,+', ',', new_attr)
                new_attr = re.sub(r'^\s*,', '', new_attr)
                new_attr = re.sub(r',\s*$', '', new_attr)
                custom_node_style = 'penwidth=3, color="red"'
                if new_attr:
                    new_attr = new_attr + ', ' + custom_node_style
                else:
                    new_attr = custom_node_style
                line = f"{node_match.group(1)}{new_attr}{suffix}"
            # Now override the fillcolor based on the node's class distribution.
            counts = clf.tree_.value[node_id][0]  # array of class counts
            total = counts.sum()
            # Check if there's a unique majority.
            if total > 0 and np.sum(counts == counts.max()) == 1:
                majority = counts.max()
                # Find the second largest count.
                second_largest = np.partition(counts, -2)[-2]
                # Compute a ratio: 0 if majority == second largest, 1 if second largest == 0.
                ratio = (majority - second_largest) / majority
            else:
                ratio = 0
            if ratio == 0:
                new_fill = "#FFFFFF"  # white
            else:
                # Get the index of the majority class.
                majority_class = int(np.argmax(counts))
                base_color = COLOUR_SEQUENCE[majority_class]
                new_fill = blend_colors("#FFFFFF", base_color, ratio)
            # Replace (or add) the fillcolor attribute.
            if 'fillcolor=' in line:
                line = re.sub(r'fillcolor="[^"]*"', f'fillcolor="{new_fill}"', line)
            else:
                line = line.rstrip("]") + f', fillcolor="{new_fill}"]'
            new_lines.append(line)
            continue
        
        # --- Process edge definitions ---
        edge_match = edge_pattern.match(line)
        if edge_match:
            src = int(edge_match.group(2))
            dst = int(edge_match.group(3))
            if (src, dst) in path_edges:
                attr_content = edge_match.group(5) if edge_match.group(5) is not None else ""
                attr_content = re.sub(r'(,\s*)?(color|penwidth)\s*=\s*("[^"]*"|[^,\]]+)', '', attr_content)
                attr_content = attr_content.strip()
                attr_content = re.sub(r'^\s*,', '', attr_content)
                attr_content = re.sub(r',\s*$', '', attr_content)
                custom_edge_style = 'color="red", penwidth=3'
                new_attr = f"{attr_content}, {custom_edge_style}" if attr_content else custom_edge_style
                new_attr_block = f' [{new_attr}]'
                line = f"{edge_match.group(1)}{edge_match.group(2)} -> {edge_match.group(3)}{new_attr_block};"
            new_lines.append(line)
            continue
        
        new_lines.append(line)
    
    return "\n".join(new_lines)


def main():
    HORIZONTAL = "../Logo.png"
    ICON = "../Icon.png"

    
    st.logo(HORIZONTAL, size="large", link="http://www.krisolis.ie", icon_image=ICON)
    st.set_page_config(
        page_title="Krisolis Supervised Learning Demonstration",
        page_icon=ICON,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': """# Krisolis Supervised Learning Demonstration. 
            Created by Eoghan Staunton @Krisolis
            Using scikit-learn and supertree"""
        })

    st.title("Supervised Learning")
    st.markdown("Use the controls in the sidebar to train a model. Once it is trained you can use it to make predictions.")
    st.header("🌳 Demonstration with Decision Trees")
    st.markdown("This app demonstrates a Supervised Machine Learning Model called a **Decision Tree Classifier** on the Iris dataset.", 
             help="""The Iris dataset is a classic and widely utilized resource in the fields of statistics and machine learning. Introduced by British statistician and biologist Ronald Fisher in his 1936 paper "The use of multiple measurements in taxonomic problems," the dataset comprises 150 samples of iris flowers, divided equally among three species: Iris setosa, Iris versicolor, and Iris virginica. Each sample includes four features: sepal length, sepal width, petal length, and petal width, all measured in centimeters. \n\n Due to its simplicity and well-structured nature, the Iris dataset has become a standard test case for various statistical classification techniques and machine learning algorithms. """)
    

    
    
    # Load the Iris dataset and create a DataFrame.
    # iris = load_iris()
    # df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # df["target"] = iris.target
    # mapping = {i: name for i, name in enumerate(iris.target_names)}
    # df["target_name"] = df["target"].map(mapping)

    df = pd.read_csv("../Supervised-Data/fraud_dataset_C_good_quality.csv")
    st.session_state["df"] = df
    mapping = {i: name for i, name in enumerate(["legitimate", "fraudulent"])}
    df["fraud_name"] = df["fraud"].map(mapping)

    # Sidebar: all user inputs.
    st.sidebar.subheader("Set maximum tree depth")
    st.session_state["max_depth"] = st.sidebar.slider("Max Depth",
                                  min_value=1, max_value=5, value=3, step=1)
    
    # Button to train the model.
    if st.sidebar.button("Train and Visualise Model"):
        clf, X, y, feature_names, class_names = train_model(st.session_state["df"],st.session_state["max_depth"])
        st.session_state["trained_model"] = clf
        st.session_state["X"] = X
        st.session_state["y"] = y
        st.session_state["feature_names"] = feature_names
        st.session_state["class_names"] = class_names
        
        # Initially, generate visualisation without a test sample.
        st.session_state["viz_html"] = generate_vis(clf, X, y, feature_names, class_names)
        st.session_state["model_trained"] = True

    # Before the model is trained, show the interactive dataset visualisation.
    if not st.session_state.get("model_trained", False):
        show_data_visualisation(df, df.columns)
        st.sidebar.info("Train a model to see the decision tree visualisation and to input a new observation.")
    else:
        # Once the model is trained, display the SuperTree visualisation.
        st.subheader(f"Trained {st.session_state["max_depth"]}-Level Decision Tree")
        st.components.v1.html(st.session_state["viz_html"], height=700)
        
        # In the sidebar, display new observation inputs.
        st.sidebar.subheader("Make a Prediction for a New Observation")
        obs_petal_length = st.sidebar.number_input("Petal Length (cm)", value=4.35, key="obs_petal_length")
        obs_petal_width  = st.sidebar.number_input("Petal Width (cm)",  value=1.3, key="obs_petal_width")
        obs_sepal_length = st.sidebar.number_input("Sepal Length (cm)", value=5.8, key="obs_sepal_length")
        obs_sepal_width  = st.sidebar.number_input("Sepal Width (cm)",  value=3.0, key="obs_sepal_width")
        
        # Only allow prediction if the model is trained.
        if st.sidebar.button("Make Prediction", key="predict_obs"):
            test_sample = np.array([obs_sepal_length, obs_sepal_width, obs_petal_length, obs_petal_width]).reshape(1, -1)
            clf = st.session_state["trained_model"]
            pred = clf.predict(test_sample)[0]
            prob = clf.predict_proba(test_sample)[0]
            st.subheader(f"🔮 Prediction Results - Predicted Species: {st.session_state['class_names'][pred].title()}")
            

            # Create a DataFrame with class names, probabilities, and colors
            df = pd.DataFrame({
                "Species": st.session_state["class_names"],
                "Probability": prob,
                "Color": COLOUR_SEQUENCE[:len(st.session_state["class_names"])]
            })

            # Sort the DataFrame by Probability in descending order
            df = df.sort_values(by="Probability", ascending=False)


            # Create a horizontal bar plot with custom colors
            fig = px.bar(df, x="Probability", y="Species", orientation="h", color="Species", 
                        color_discrete_map={cls: color for cls, color in zip(st.session_state["class_names"], COLOUR_SEQUENCE)})

            # Update the layout to display the probabilities with two decimal places
            fig.update_layout(yaxis=dict(tickfont=dict(size=14)), xaxis=dict(tickformat=".2f"), legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1))


            # Create columns to display multiple visualizations
            col1, col2 = st.columns([2, 2])
            
            # Generate the modified DOT with highlighted nodes and edges.
            highlighted_dot = highlight_dot(clf, test_sample, st.session_state["feature_names"], st.session_state['class_names'])

            # Display the bar chart in the first column
            with col1:
                # Display the plot in the Streamlit app
                st.write("**📊 Species Probabilities:**")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.write("**🛤️ Decision Path:**")
                st.graphviz_chart(highlighted_dot)

            

if __name__ == "__main__":
    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False
    main()
