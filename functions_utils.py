from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_hex
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from functions_utils import *
from scipy.stats import zscore
from collections import defaultdict
from matplotlib.lines import Line2D

def cluster_visual(df, k=3, feature_option=None, summary_df=None):
    
    df_clusterd = df.copy()
    squad_names = df_clusterd['squad'] if 'squad' in df_clusterd.columns else None
    X = df_clusterd.select_dtypes(include='number')

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df_clusterd['cluster'] = clusters
    if squad_names is not None:
        df_clusterd['squad'] = squad_names

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Build cluster name map from summary_df
    if summary_df is not None:
        if 'cluster' in summary_df.columns:
            cluster_name_map = summary_df.set_index('cluster')['Cluster Label'].to_dict()
        else:
            # Assume index is the cluster number
            cluster_name_map = summary_df['Cluster Label'].to_dict()
    else:
        # Fallback: generic labels
        cluster_name_map = {i: f"Cluster {i}" for i in range(k)}

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=clusters, cmap='Set1', s=80, edgecolor='k'
    )

    # Add team names
    if squad_names is not None:
        for i, name in enumerate(squad_names):
            ax.text(
                X_pca[i, 0], X_pca[i, 1], name,
                fontsize=8, color='black',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
            )

    # Ellipses around clusters
    unique_clusters = np.unique(clusters)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
    for cluster_id, color in zip(unique_clusters, colors):
        cluster_points = X_pca[clusters == cluster_id]
        center = cluster_points.mean(axis=0)
        std_dev = cluster_points.std(axis=0)

        ellipse = Ellipse(
            xy=center,
            width=std_dev[0]*4,
            height=std_dev[1]*4,
            edgecolor=color,
            facecolor='none',
            linewidth=2,
            linestyle='--'
        )
        ax.add_patch(ellipse)

    # Correct legend labels
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', color=colors[i],
                   label=cluster_name_map.get(i, f"Cluster {i}"))
        for i in unique_clusters
    ]

    legend = ax.legend(handles=handles, title='Clusters', loc='upper left', bbox_to_anchor=(1.05, 1), facecolor='white')
    for text in legend.get_texts():
        text.set_color("black")
    legend.get_title().set_color("black")

    ax.set_title('Team Clusters', fontsize=16, color='black')
    ax.tick_params(colors='black')
    ax.spines[:].set_color('black')
    ax.grid(True, color='gray', alpha=0.3)
    ax.set_xlabel("PCA 1", color='black')
    ax.set_ylabel("PCA 2", color='black')

    plt.tight_layout()
    return fig



def clusterd_dataframe(df, k=3, max_k=10, plot_elbow=True):
    df_clusterd = df.copy()
    squad_names = df_clusterd['squad'] if 'squad' in df_clusterd.columns else None
    X = df_clusterd.select_dtypes(include='number')

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply k-means with specified k
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df_clusterd['cluster'] = clusters
    if squad_names is not None:
        df_clusterd['squad'] = squad_names

    return df_clusterd



def find_optimal_k(df, max_k=10, plot=True):
    # Select numerical features from the dataframe
    X = df.select_dtypes(include='number')

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    K_range = range(1, max_k + 1)

    # Calculate inertia for each k in the range
    for k_val in K_range:
        kmeans = KMeans(n_clusters=k_val, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # Plot elbow curve
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, inertias, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Find the optimal number of clusters')
        plt.grid(True)
        plt.show()
    


def cluster_summerise(df, k=3, max_k=10, plot_elbow=True):
    df_clusterd = df.copy()
    squad_names = df_clusterd['squad'] if 'squad' in df_clusterd.columns else None
    X = df_clusterd.select_dtypes(include='number')

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply k-means with specified k
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df_clusterd['cluster'] = clusters
    if squad_names is not None:
        df_clusterd['squad'] = squad_names

    # Show cluster summary
    cluster_summary = df_clusterd.groupby('cluster').mean(numeric_only=True)

    return cluster_summary



def cluster_visual_3d_interactive(df, k=3, feature_option=None, summary_df=None):
    df_clusterd = df.copy()
    squad_names = df_clusterd['squad'] if 'squad' in df_clusterd.columns else None
    X = df_clusterd.select_dtypes(include='number')

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df_clusterd['cluster'] = clusters
    if squad_names is not None:
        df_clusterd['squad'] = squad_names

    # Use name_clusters to label the clusters
    df_named = name_clusters(df_clusterd.copy(), feature_option=feature_option)

    # Build the cluster name map from summary_df (as done in the 2D plot)
    if summary_df is not None:
        if 'cluster' in summary_df.columns:
            cluster_name_map = summary_df.set_index('cluster')['Cluster Label'].to_dict()
        else:
            # Assume index is the cluster number
            cluster_name_map = summary_df['Cluster Label'].to_dict()
    else:
        # Fallback: generic labels
        cluster_name_map = {i: f"Cluster {i}" for i in range(k)}

    # Apply PCA for 3D visualization
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # Color mapping for clusters
    cmap = cm.get_cmap('Set1', k)
    cluster_colors = [to_hex(cmap(i)) for i in range(k)]

    # Create the 3D Plotly figure
    fig = go.Figure()

    for cluster_id in np.unique(clusters):
        cluster_points = X_pca[clusters == cluster_id]
        hover_text = df_clusterd.loc[clusters == cluster_id, 'squad'] if squad_names is not None else None
        # Get the label name for the cluster from the cluster_name_map
        label_name = cluster_name_map.get(cluster_id, f"Cluster {cluster_id}")

        fig.add_trace(go.Scatter3d(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            z=cluster_points[:, 2],
            mode='markers+text',
            marker=dict(
                size=6,
                color=cluster_colors[cluster_id],
                line=dict(width=1, color='black')
            ),
            text=hover_text,
            name=label_name,
            textposition="top center"
        ))

        # Optional translucent sphere around the cluster
        centroid = cluster_points.mean(axis=0)
        radius = np.linalg.norm(cluster_points - centroid, axis=1).mean()

        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_sphere = centroid[0] + radius * np.cos(u) * np.sin(v)
        y_sphere = centroid[1] + radius * np.sin(u) * np.sin(v)
        z_sphere = centroid[2] + radius * np.cos(v)

        fig.add_trace(go.Surface(
            x=x_sphere,
            y=y_sphere,
            z=z_sphere,
            showscale=False,
            opacity=0.1,
            surfacecolor=np.ones_like(x_sphere),
            colorscale=[[0, cluster_colors[cluster_id]], [1, cluster_colors[cluster_id]]],
            hoverinfo='skip',
            showlegend=False
        ))
        
    fig.update_layout(
        title="Interactive Visualization",  # Title text
        title_x=0,  # Position the title at the center horizontally
        title_y=0.98,  # Position the title slightly below the top
        title_font=dict(
            size=18,  # Font size for the title
            color='white',  # Font color for the title
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        scene=dict(
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,  # Remove the zero line (axis line)
                linecolor='rgba(255, 255, 255, 0.2)',  # Make the axis lines very faint
                linewidth=0.5  # Set the border thickness around the plot
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,  # Remove the zero line (axis line)
                linecolor='rgba(255, 255, 255, 0.2)',  # Make the axis lines very faint
                linewidth=0.5  # Set the border thickness around the plot
            ),
            zaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,  # Remove the zero line (axis line)
                linecolor='rgba(255, 255, 255, 0.2)',  # Make the axis lines very faint
                linewidth=0.5  # Set the border thickness around the plot
            ),
            bgcolor='#343541'  # Set the background color of the scene
        ),
        paper_bgcolor='#1e1e1e',  # Set the paper background color
        font=dict(color='white'),
        width=1200,
        height=800,
        showlegend=False,  # Ensure the legend is visible
        legend=dict(
            x=1,  # Position the legend in the center horizontally
            y=1.15,  # Position the legend slightly above the graph
            orientation="h",  # Set horizontal orientation
            font=dict(size=12, color='white'),
            bgcolor='rgba(0,0,0,0)',  # Make legend background transparent
            bordercolor='rgba(0,0,0,0)',  # Remove border color
            borderwidth=0,  # Remove the border width around the legend
            itemwidth=30,  # Optionally control the width of legend items
            traceorder='normal',  # Ensure trace order remains consistent
            itemsizing='constant'  # Ensure legend items are constant in size
        )
    )



    return fig



def get_cluster_descriptions(summary_df):
    label_descriptions = {
        # Attacking
        "Elite Attack": "High shots, xG and goal output.",
        "Clinical Finishers": "Score more than expected with fewer chances.",
        "Wasteful Finishers": "High xG, low goals — poor finishing.",
        "Low Threat Teams": "Few shots, low xG and goals.",
        "Slight xG Over-Performance": "Efficient with slightly better than expected goals.",
        "Slight xG Under-Performance": "Generate chances but struggle to convert.",
        "Shot-Heavy, Low Conversion": "Take many shots but few goals.",
        "Underwhelming Attackers": "Low in both volume and efficiency.",
        "Steady but Unremarkable": "Average in volume and finishing.",
        "Balanced Attackers": "Solid but not standout attacking profile.",

        # Defensive
        "Active, Conceed Little": "Concede little, active in defense.",
        "Elite Protection": "Rarely tested and concede few goals.",
        "Active, Conceed Many": "Face heavy pressure and concede often.",
        "Passive & Leaky": "Low activity and concede heavily.",
        "Busy Backline": "Active defense but still concede.",
        "Passive but Effective": "Low action but decent goal prevention.",
        "Active Defenders": "Involved frequently, varying success.",
        "Average Defenders": "Mid-level across defensive metrics.",

        # Possession & Passing
        "Elite Possession Teams": "High possession, passing, and territory control.",
        "Slow, Safe Possession": "Hold the ball but lack penetration.",
        "Direct, Low-Possession": "Quick transitions with little buildup.",
        "Territorial Without Penetration": "Attack territory but don't break lines.",
        "Progressive but Direct": "Advance ball quickly without sustained possession.",
        "Possession-Oriented": "Comfortable with ball and build-up.",
        "Balanced Possession": "Average possession and progression.",
        "Mixed Style Teams": "No clear stylistic identity.",
    }

    unique_labels = summary_df['Cluster Label'].unique()
    filtered_descriptions = {label: desc for label, desc in label_descriptions.items() if label in unique_labels}

    return filtered_descriptions


def name_clusters(summary_df, feature_option=None):
    from scipy.stats import zscore
    import numpy as np

    # Only apply zscore to numeric columns
    numeric_cols = summary_df.select_dtypes(include='number').columns
    z_df = summary_df[numeric_cols].apply(zscore)

    if feature_option == "Attacking":
        gls = summary_df['per_90_minutes_gls']
        xg = summary_df['per_90_minutes_xg']
        sh = summary_df['standard_sh/90']
        sot = summary_df['standard_sot/90']

        gls_75, gls_25 = np.percentile(gls, [75, 25])
        xg_75, xg_25 = np.percentile(xg, [75, 25])
        sh_75, sh_25 = np.percentile(sh, [75, 25])
        gls_median = np.median(gls)
        xg_median = np.median(xg)
        sh_median = np.median(sh)

        def classify_team(g, x, s, so):
            if g >= gls_75 and x >= xg_75 and s >= sh_75:
                return "Elite Attack"
            elif g >= gls_75 and x < xg_25:
                return "Clinical Finishers"
            elif g < gls_25 and x >= xg_75:
                return "Wasteful Finishers"
            elif g < gls_25 and x < xg_25 and s < sh_25:
                return "Low Threat Teams"
            else:
                if g > gls_median and x < xg_median:
                    return "Slight xG Over-Performance"
                elif x > xg_median and g < gls_median:
                    return "Slight xG Under-Performace"
                elif s > sh_median and g < gls_median:
                    return "Shot-Heavy, Low Conversion"
                elif g < gls_median and x < xg_median:
                    return "Underwhelming Attackers"
                elif abs(g - gls_median) < 0.1 and abs(x - xg_median) < 0.1:
                    return "Steady but Unremarkable"
                else:
                    return "Balanced Attackers"

        summary_df['Cluster Label'] = [classify_team(g, x, s, so) for g, x, s, so in zip(gls, xg, sh, sot)]

    elif feature_option == "Defensive":
        ga = summary_df['performance_ga90']
        sota = summary_df['performance_sota90']
        defensive_actions = summary_df[['blocks_blocks90', 'tkl+int90', 'clr90']].mean(axis=1)

        ga_25, ga_75 = np.percentile(ga, [25, 75])
        sota_75 = np.percentile(sota, 75)
        actions_75 = np.percentile(defensive_actions, 75)
        actions_25 = np.percentile(defensive_actions, 25)
        ga_median = np.median(ga)
        sota_median = np.median(sota)
        actions_median = np.median(defensive_actions)

        def classify_defense(g, s, a):
            if g <= ga_25 and a >= actions_75:
                return "Active, Conceed Little"
            elif g <= ga_25 and s <= sota_median:
                return "Elite Protection"
            elif g >= ga_75 and s >= sota_75:
                return "Active, Conceed Many"
            elif g >= ga_75 and a <= actions_25:
                return "Passive & Leaky"
            elif a >= actions_75 and g >= ga_median:
                return "Busy Backline"
            elif a < actions_median and g < ga_median:
                return "Passive but Effective"
            elif a >= actions_median:
                return "Active Defenders"
            else:
                return "Average Defenders"

        summary_df['Cluster Label'] = [classify_defense(g, s, a) for g, s, a in zip(ga, sota, defensive_actions)]

    elif feature_option == "Possession & Passing":
        poss = summary_df['poss']
        touches = summary_df['touches_att_3rd90']
        pass_quality = summary_df[['total_cmp%', 'prgp90']].mean(axis=1)

        poss_75, poss_25 = np.percentile(poss, [75, 25])
        touches_75, touches_25 = np.percentile(touches, [75, 25])
        passq_75, passq_25 = np.percentile(pass_quality, [75, 25])
        poss_median = np.median(poss)
        passq_median = np.median(pass_quality)
        touches_median = np.median(touches)

        def classify_possession(p, t, q):
            if p >= poss_75 and q >= passq_75 and t >= touches_75:
                return "Elite Possession Teams"
            elif p >= poss_75 and q < passq_25:
                return "Slow, Safe Possession"
            elif p < poss_25 and q < passq_25:
                return "Direct, Low-Possession"
            elif t >= touches_75 and q < passq_median:
                return "Territorial Without Penetration"
            elif q >= passq_75 and p < poss_median:
                return "Progressive but Direct"
            elif p >= poss_median and q >= passq_median:
                return "Possession-Oriented"
            elif abs(p - poss_median) < 0.1 and abs(q - passq_median) < 0.1:
                return "Balanced Possession"
            else:
                return "Mixed Style Teams"

        summary_df['Cluster Label'] = [classify_possession(p, t, q) for p, t, q in zip(poss, touches, pass_quality)]

    else:
        summary_df['Cluster Label'] = [f"Cluster {i}" for i in summary_df.index]

    return summary_df
