import streamlit as st

st.title("🏀 Interactive NBA Data Analysis and Visualization")
st.markdown("""
## Welcome to the NBA Dashboard!  
⬅️ Use the sidebar to explore visualizations and analytics tools:

- 🎯 **Shot Chart Visualizer**  
  Explore where your favorite NBA players are taking their shots and how efficient they are across the court.  
  - View individual player shot maps  
  - Compare shooting zones (e.g., paint, midrange, 3PT)  
  - Filter by team, season, or game
            
- ⛹️‍♂️ **Player Clustering Analysis**
  Redefine player roles on the basketball court using unsupervised machine learning.
  - Map players based on play-type frequency, average shot distance, and per 36 minute statistics
  - Tune how many clusters to apply to the machine learning algorithm
  - Choose between PCA, tSNE, and UMAP dimensionality reduction for player mapping
  - See which features stand out in each cluster, in order to contextualize Kmeans and clustering results
""")


