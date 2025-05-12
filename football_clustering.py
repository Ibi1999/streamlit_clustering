import streamlit as st
import pandas as pd
from functions_utils import *
import plotly.express as px
from scipy.stats import zscore

# Set wide layout
st.set_page_config(layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("⚽ Football Data\nMachine Learning Clustering")

    # Dropdown-style expander for how-to instructions
    with st.expander("ℹ️ How to use", expanded=False):
        st.markdown("""
        ### How to use this app:
        1. **Select the competition** (e.g., Champions League or Premier League).
        2. **Choose a feature set** (Attacking, Defensive, Possession & Passing).
        3. **Pick the number of clusters (k)** using the slider.
        4. Click **Run Clustering** to generate:
            - Visualizations (2D & 3D)
            - Cluster summaries
            - A detailed output dataframe.
        """)

    # Cluster Descriptions moved back here
    with st.expander("📊 Cluster Descriptions", expanded=False):
        feature_option_sidebar = st.session_state.get("feature_option_sidebar", "Attacking")

        if feature_option_sidebar == "Attacking":
            st.markdown("### 🟥 Attacking Clusters")
            attacking_labels = {
                "Elite Attack": "High shots, xG and goal output.",
                "Clinical Finishers": "Score more than expected with fewer chances.",
                "Wasteful Finishers": "High xG, low goals — poor finishing.",
                "Low Threat Teams": "Few shots, low xG and goals.",
                "Slight xG Over-Performance": "Efficient with slightly better than expected goals.",
                "Slight xG Under-Performance": "Generate chances but struggle to convert.",
                "Shot-Heavy, Low Conversion": "Take many shots but few goals.",
                "Underwhelming Attackers": "Low in both volume and efficiency.",
                "Steady but Unremarkable": "Average in volume and finishing.",
                "Balanced Attackers": "Solid but not standout attacking profile."
            }
            for label, desc in attacking_labels.items():
                st.write(f"**{label}:** {desc}")

        elif feature_option_sidebar == "Defensive":
            st.markdown("### 🟦 Defensive Clusters")
            defensive_labels = {
                "Active, Conceed Little": "Concede little, active in defense.",
                "Elite Protection": "Rarely tested and concede few goals.",
                "Active, Conceed Many": "Face heavy pressure and concede often.",
                "Passive & Leaky": "Low activity and concede heavily.",
                "Busy Backline": "Active defense but still concede.",
                "Passive but Effective": "Low action but decent goal prevention.",
                "Active Defenders": "Involved frequently, varying success.",
                "Average Defenders": "Mid-level across defensive metrics."
            }
            for label, desc in defensive_labels.items():
                st.write(f"**{label}:** {desc}")

        elif feature_option_sidebar == "Possession & Passing":
            st.markdown("### 🟩 Possession & Passing Clusters")
            possession_labels = {
                "Elite Possession Teams": "High possession, passing, and territory control.",
                "Slow, Safe Possession": "Hold the ball but lack penetration.",
                "Direct, Low-Possession": "Quick transitions with little buildup.",
                "Territorial Without Penetration": "Attack territory but don't break lines.",
                "Progressive but Direct": "Advance ball quickly without sustained possession.",
                "Possession-Oriented": "Comfortable with ball and build-up.",
                "Balanced Possession": "Average possession and progression.",
                "Mixed Style Teams": "No clear stylistic identity."
            }
            for label, desc in possession_labels.items():
                st.write(f"**{label}:** {desc}")

# --- Title ---
st.title("Team Clustering")

# --- Inputs ---
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    feature_option = st.selectbox(
        "Select Feature Set for Clustering:",
        ("Attacking", "Defensive", "Possession & Passing")
    )
    st.session_state["feature_option_sidebar"] = feature_option  # update for sidebar display

with col3:
    k = st.slider("Select number of clusters (k):", 1, 10, 4)

with col1:
    competition_option = st.selectbox(
        "Select Competition:",
        ("UEFA Champions League 2024/2025", "Premier League 2024/2025")
    )

# --- Load data ---
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

# --- Run Clustering ---
if st.button("Run Clustering"):
    summary_df = cluster_summerise(selected_df, k)
    summary_df = name_clusters(summary_df, feature_option=feature_option)

    dataframe_output = clusterd_dataframe(selected_df, k)
    cluster_label_map = summary_df['Cluster Label'].to_dict()
    dataframe_output['Cluster Label'] = dataframe_output['cluster'].map(cluster_label_map)

    fig2d = cluster_visual(selected_df, k, feature_option=feature_option, summary_df=summary_df)
    fig3d = cluster_visual_3d_interactive(selected_df, k, feature_option=feature_option, summary_df=summary_df)
    cluster_descriptions = get_cluster_descriptions(summary_df)

    # Save to session_state
    st.session_state["summary_df"] = summary_df
    st.session_state["dataframe_output"] = dataframe_output
    st.session_state["fig2d"] = fig2d
    st.session_state["fig3d"] = fig3d
    st.session_state["cluster_descriptions"] = cluster_descriptions

# --- Display Results ---
if "summary_df" in st.session_state and st.session_state["summary_df"] is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2D Cluster model")
        st.pyplot(st.session_state["fig2d"], use_container_width=True)

    with col2:
        st.subheader("3D Cluster model")
        st.plotly_chart(st.session_state["fig3d"], use_container_width=True)

    st.subheader("Cluster Summary")
    st.dataframe(st.session_state["summary_df"])

    st.subheader("Model Output Dataframe")
    st.dataframe(st.session_state["dataframe_output"])
