import streamlit as st
import pandas as pd
from functions_utils import *
import plotly.express as px
from scipy.stats import zscore

# Set wide layout
st.set_page_config(layout="wide")

# Title
st.title("Football Clustering using Machine Learning")

# Create three columns for the dropdowns and slider
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    feature_option = st.selectbox(
        "Select Feature Set for Clustering:",
        ("Attacking", "Defensive", "Possession & Passing")
    )

with col3:
    k = st.slider("Select number of clusters (k):", 1, 10, 4)

with col1:
    competition_option = st.selectbox(
        "Select Competition:",
        ("UEFA Champions League 2024/2025", "Premier League 2024/2025")
    )

# Load the corresponding dataframe based on feature and competition
@st.cache_data
def load_dataframe(feature_option, competition_option):
    file_map = {
        "UEFA Champions League 2024/2025": {
            "Attacking": "attacking_features_df.pkl",
            "Defensive": "defensive_features_df.pkl",
            "Possession & Passing": "possession_passing_features_df.pkl"
        },
        "Premier League 2024/2025": {
            "Attacking": "attacking_features_pl_df.pkl",
            "Defensive": "defensive_features_pl_df.pkl",
            "Possession & Passing": "possession_passing_features_pl_df.pkl"
        }
    }
    return pd.read_pickle(file_map[competition_option][feature_option])

selected_df = load_dataframe(feature_option, competition_option)

# Run clustering and show plot
if st.button("Run Clustering"):
    summary_df = cluster_summerise(selected_df, k)
    summary_df = name_clusters(summary_df, feature_option=feature_option)  # This now matches the function's parameter name

    dataframe_output = clusterd_dataframe(selected_df, k)

    # Merge cluster labels into output dataframe
    cluster_label_map = summary_df['Cluster Label'].to_dict()
    dataframe_output['Cluster Label'] = dataframe_output['cluster'].map(cluster_label_map)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2D Cluster model")
        fig = cluster_visual(selected_df, k, feature_option=feature_option, summary_df=summary_df)
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.subheader("3D Cluster model")
        fig3d = cluster_visual_3d_interactive(selected_df, k, feature_option=feature_option, summary_df=summary_df)
        st.plotly_chart(fig3d, use_container_width=True)
    

    st.subheader("Cluster Descriptions")
    cluster_descriptions = get_cluster_descriptions(summary_df)
    for label, description in cluster_descriptions.items():
        st.write(f"**{label}:** {description}")

    st.subheader("Cluster Summary")
    st.dataframe(summary_df)

    # Display cluster summary as it was before
    st.subheader("Model Output Dataframe")
    st.dataframe(dataframe_output)
