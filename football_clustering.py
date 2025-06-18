import streamlit as st
import pandas as pd
from functions_utils import *
import plotly.express as px
from scipy.stats import zscore

st.set_page_config(layout="wide")

# --- Sidebar ---
with st.sidebar:
    # Load and display logo in the top-left
    from PIL import Image
    logo = Image.open("logo.png")
    st.image(logo, width=80)  # Adjust width as needed
    st.markdown("<small>Created by Ibrahim Oksuzoglu</small>", unsafe_allow_html=True)
    st.title("🤖 Machine learning")
    clustering_type = st.selectbox("Select Clustering Type:", ("Team Clustering", "Player Clustering"))

    with st.expander("🤔 What this app does", expanded=False): 
        st.markdown("""
        ### What this app does:
        1) This app clusters football teams based on their playing styles (Attacking, Defensive, Possession & Passing), helping identify teams with similar strategies and performance trends across competitions.
        
        2) It also clusters players based on strengths like Defensive, Pass Types, Creativity, and Goal Scoring, allowing comparisons and analysis of similar player profiles accross competitions
        """)


    with st.expander("ℹ️ How to use", expanded=True):
        if clustering_type == "Team Clustering":
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
        else:
            st.markdown("""
            ### How to use this app:
            1. **Select the competition** (e.g., Champions League or Premier League).
            2. **Choose a player** and select one of their valid positions.
            3. **Pick a feature set** (e.g., Goalkeeping, Passing, Creativity).
            4. **Select number of clusters (k)** and click **Run Clustering**.
            5. The app will display:
                - Players in the same cluster (or all clusters if selected).
                - Visualizations (2D & 3D).
                - Cluster summaries and a detailed output.
            """)

    # Only show descriptions if "Team Clustering" is selected
    if clustering_type == "Team Clustering":
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

# --- TEAM CLUSTERING ---
if clustering_type == "Team Clustering":
    st.title("🧩 Football Team Clustering")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        feature_option = st.selectbox("🧠 Select Feature Set", ("Attacking", "Defensive", "Possession & Passing"))
        st.session_state["feature_option_sidebar"] = feature_option

    with col3:
        k = st.slider("🎯 Select number of clusters (k):", 1, 10, 4)

    with col1:
        competition_option = st.selectbox(
            "🌍 Select Competition:",
            ("UEFA Champions League 2024/2025", "Premier League 2024/2025")
        )

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

    if st.button("Run Clustering"):
        summary_df = cluster_summerise(selected_df, k)
        summary_df = name_clusters(summary_df, feature_option=feature_option)

        dataframe_output = clusterd_dataframe(selected_df, k)
        cluster_label_map = summary_df['Cluster Label'].to_dict()
        dataframe_output['Cluster Label'] = dataframe_output['cluster'].map(cluster_label_map)

        fig2d = cluster_visual(selected_df, k, feature_option=feature_option, summary_df=summary_df)
        fig3d = cluster_visual_3d_interactive(selected_df, k, feature_option=feature_option, summary_df=summary_df)
        cluster_descriptions = get_cluster_descriptions(summary_df)

        st.session_state.update({
            "summary_df": summary_df,
            "dataframe_output": dataframe_output,
            "fig2d": fig2d,
            "fig3d": fig3d,
            "cluster_descriptions": cluster_descriptions
        })

    if "summary_df" in st.session_state:
        st.markdown("---")
        st.success(f"Clustering Teams based on **{feature_option}** features")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2D Cluster model")
            st.pyplot(st.session_state["fig2d"], use_container_width=True)
        with col2:
            st.subheader("3D Cluster model")
            st.plotly_chart(st.session_state["fig3d"], use_container_width=True)

        st.markdown("---")
        st.subheader("🧮 Cluster Summary")
        st.dataframe(st.session_state["summary_df"])

        st.markdown("---")
        st.subheader("📋 Model Output")
        st.dataframe(st.session_state["dataframe_output"])

# --- PLAYER CLUSTERING ---
else:
    st.title("👤 Football Player Clustering")

    @st.cache_data
    def load_player_info(competition):
        path = "player_names_pl.pkl" if competition == "Premier League 2024/2025" else "player_names.pkl"
        df = pd.read_pickle(path)
        if 'player_id' not in df.columns or 'player' not in df.columns:
            st.warning(f"⚠️ player_names file at `{path}` is missing required columns.")
        return df

    @st.cache_data
    def load_player_dataframe(feature, competition):
        file_map = {
            "UEFA Champions League 2024/2025": {
                "Goalkeeping": "goalkeeping_player_features.pkl",
                "Defensive": "defensive_player_features.pkl",
                "Pass Types": "pass_type_player_features.pkl",
                "Creativity": "creative_player_features_df.pkl",
                "Goal Scoring": "goal_scoring_player_features.pkl"
            },
            "Premier League 2024/2025": {
                "Goalkeeping": "goalkeeping_player_features_pl.pkl",
                "Defensive": "defensive_player_features_pl.pkl",
                "Pass Types": "pass_type_player_features_pl.pkl",
                "Creativity": "creative_player_features_df_pl.pkl",
                "Goal Scoring": "goal_scoring_player_features_pl.pkl"
            }
        }
        return pd.read_pickle(file_map[competition][feature])

    player_competition_option = st.selectbox(
        "🌍 Select Competition:",
        ("UEFA Champions League 2024/2025", "Premier League 2024/2025"),
        key="player_competition"
    )

    player_info_df = load_player_info(player_competition_option)
    player_list = player_info_df["player"].sort_values().unique().tolist()

    col1, col2 = st.columns([1, 1])
    with col1:
        selected_player = st.selectbox("🔍 Select Player:", options=player_list, index=0)

    with col2:
        row = player_info_df[player_info_df["player"] == selected_player]
        position_string = row["pos"].values[0] if not row.empty else ""
        position_options = [p.strip() for p in position_string.split(",")] if position_string else ["GK", "DF", "MF", "FW"]
        selected_position = st.selectbox("📌 Select Position:", options=position_options)

    col3, col4 = st.columns([1, 1])
    with col3:
        if selected_position == "GK":
            feature_options = ["Goalkeeping", "Pass Types"]
        else:
            feature_options = ["Defensive", "Pass Types", "Creativity", "Goal Scoring"]

        player_feature_option = st.selectbox(
            "🧠 Select Feature Set for Clustering:",
            options=feature_options,
            key="player_feature_option"
        )
        st.session_state["feature_option_sidebar"] = player_feature_option

    with col4:
        player_k = st.slider("🎯 Select number of clusters (k):", 1, 20, 20, key="player_k")

    player_df = load_player_dataframe(player_feature_option, player_competition_option)

    col1, col2, _ = st.columns([1, 2, 7])
    with col1:
        run_clustering = st.button("Run Clustering")
    with col2:
        show_all_clusters = st.checkbox("Show all clusters (Experimental)", value=False)

    if run_clustering:
        st.markdown("### 📊 Player Cluster Visualisations")

        fig_2d = cluster_player_visual(
            df=player_df,
            k=player_k,
            summary_df=None,
            pos=selected_position,
            name=selected_player if not show_all_clusters else None
        )

        if fig_2d:
            fig_3d = cluster_player_visual_3d_interactive(
                df=player_df,
                k=player_k,
                summary_df=None,
                pos=selected_position,
                name=selected_player if not show_all_clusters else None
            )

            st.success(f"Showing players similar to **{selected_player}** at **{selected_position}**.")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("2D Cluster model")
                st.pyplot(fig_2d)
            with col2:
                st.subheader("3D Cluster model")
                st.plotly_chart(fig_3d, use_container_width=True)

            st.markdown("---")
            st.subheader("🧮 Cluster Summary")
            summary = cluster_player_summerise(
                player_df, 
                player_names=player_info_df, 
                k=player_k, 
                pos=selected_position, 
                name=selected_player if not show_all_clusters else None
            )
            if summary is not None and not summary.empty:
                st.dataframe(summary.style.format(precision=2), use_container_width=True)

            st.markdown("---")
            st.subheader("📋 Model Output")
            output = clusterd_player_dataframe(
                player_df, 
                player_names=player_info_df, 
                k=player_k, 
                pos=selected_position, 
                name=selected_player if not show_all_clusters else None
            )
            if output is not None and not output.empty:
                st.dataframe(output.style.format(precision=2), use_container_width=True)

        else:
            st.warning(f"No players found matching '{selected_player}' at position **{selected_position}**.")
