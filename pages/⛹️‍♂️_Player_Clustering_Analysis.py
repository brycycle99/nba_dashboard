import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import seaborn as sns
from scipy.stats import zscore
import matplotlib.pyplot as plt

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/play_style.csv", index_col=0)
    return df

df = load_data()

st.title("NBA Player Clustering: PCA, t-SNE, and UMAP")

st.write("Explore NBA players using dimensionality reduction techniques and KMeans clustering. Use the slider to select the number of groups for kmeans clustering, use the interactive plot to explore individual player stats, and then contextualize the results with average stats per cluster using the heatmap.")

features = [col for col in df.columns if col != 'PLAYER_NAME']
num_clusters = st.slider("Number of Clusters (KMeans)", min_value=2, max_value=10, value=4)

# Standardize features
X = df[features]
X_scaled = StandardScaler().fit_transform(X)

# Fit PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Fit t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# Fit UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

# Run KMeans (on PCA for consistency, can be changed)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(X_pca)
df['Cluster'] = clusters.astype(str)

# Build main dataframe
df_with_coords = pd.concat([
    df,
    pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=df.index),
    pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'], index=df.index),
    pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'], index=df.index),
], axis=1)

# Tabs for visualizations
tab1, tab2, tab3 = st.tabs(["PCA", "t-SNE", "UMAP"])

with tab1:
    st.subheader("PCA Projection")
    fig_pca = px.scatter(
        df_with_coords,
        x='PC1',
        y='PC2',
        color='Cluster',
        hover_name='PLAYER_NAME',
        hover_data={
            'PC1': False, 'PC2': False,
            'PTS_PER36': True, 'REB_PER36': True,
            'AST_PER36': True, 'BLKA_PER36': True,
            'STL_PER36': True, 'AVG_SHOT_DISTANCE': True,
            'Cluster': True
        },
        color_discrete_sequence=px.colors.qualitative.Set1,
        width=850, height=600
    )
    st.plotly_chart(fig_pca, use_container_width=True)

with tab2:
    st.subheader("t-SNE Projection")
    fig_tsne = px.scatter(
        df_with_coords,
        x='TSNE1',
        y='TSNE2',
        color='Cluster',
        hover_name='PLAYER_NAME',
        hover_data={
            'TSNE1': False, 'TSNE2': False,
            'PTS_PER36': True, 'REB_PER36': True,
            'AST_PER36': True, 'BLKA_PER36': True,
            'STL_PER36': True, 'AVG_SHOT_DISTANCE': True,
            'Cluster': True
        },
        color_discrete_sequence=px.colors.qualitative.Set1,
        width=850, height=600
    )
    st.plotly_chart(fig_tsne, use_container_width=True)

with tab3:
    st.subheader("UMAP Projection")
    fig_umap = px.scatter(
        df_with_coords,
        x='UMAP1',
        y='UMAP2',
        color='Cluster',
        hover_name='PLAYER_NAME',
        hover_data={
            'UMAP1': False, 'UMAP2': False,
            'PTS_PER36': True, 'REB_PER36': True,
            'AST_PER36': True, 'BLKA_PER36': True,
            'STL_PER36': True, 'AVG_SHOT_DISTANCE': True,
            'Cluster': True
        },
        color_discrete_sequence=px.colors.qualitative.Set1,
        width=850, height=600
    )
    st.plotly_chart(fig_umap, use_container_width=True)

# Cluster Heatmap
st.subheader("Cluster Feature Profiles")
cluster_summary = df.copy()
cluster_summary['Cluster'] = clusters
cluster_summary = cluster_summary.groupby('Cluster')[features].mean()
cluster_summary_z = cluster_summary.apply(zscore, axis=0)

fig2, ax = plt.subplots(figsize=(18, 16))
sns.heatmap(cluster_summary_z.T, cmap='coolwarm', center=0, ax=ax)
ax.set_title("Standardized Feature Values by Cluster (Z-scored)", fontsize=14)
ax.set_ylabel("Feature")
ax.set_xlabel("Cluster")
st.pyplot(fig2)

# Download
st.subheader("Download full dataset")
csv = df_with_coords.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download CSV",
    data=csv,
    file_name='play_style_all_coords_clusters.csv',
    mime='text/csv'
)
