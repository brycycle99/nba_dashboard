# 🏀 NBA Analytics Dashboard (Streamlit)

Welcome to the **NBA Analytics Dashboard**, a Streamlit-powered platform for interactive visual exploration of NBA data.

This dashboard will host multiple data analysis tools and visualizations related to NBA player performance, team trends, and league-wide insights.

---

## 📌 Current Apps

### 1. 🔥 Shot Chart Visualizer

Visualize shot attempt densities on an interactive NBA court, using data fetched from the official NBA Stats API. Explore how different players, teams, and the league overall shoot from various areas on the court.

#### Features:
- 📅 **Season Selector**: Choose from NBA seasons 1996–2024  
- 🏆 **Season Type**: Regular Season or Playoffs  
- 🧍 **Modes**:
  - **League-wide**: Aggregate shot attempts for all players
  - **Team**: Filter shot data by NBA team
  - **Player**: Zoom in on individual player shot charts
- 🧊 **Hexbin Shot Map**: View shot frequency with hover details  
- 🏟️ **Custom Half-Court**: Accurately scaled court background using Plotly

---

## 🚧 Upcoming Apps

Planned additions to the dashboard include:

- 📈 **Player Efficiency Analyzer**  
- 🧮 **Team Stats Comparisons (Radar Charts, Trends)**  
- 🔁 **Trade Analyzer / Impact Visualizer**  
- 📊 **Game-by-Game Heatmaps**  
- 🧠 **Machine Learning Player Clustering (K-Means, t-SNE)**

Stay tuned as new tools roll out!

---

## 🧪 Requirements

- `streamlit`
- `nba_api`
- `numpy`
- `plotly`
- `matplotlib`

## 🚀 How to Run

### On Streamlit Cloud (Recommended)

Just click the **"Deploy to Streamlit"** button or upload this repo to [https://streamlit.io/cloud](https://streamlit.io/cloud). It will automatically detect `streamlit_app.py` and install dependencies from `requirements.txt`.

### Run Locally

```bash
# Step 1: Clone the repo
git clone https://github.com/your-username/nba_dashboard.git
cd nba_dashboard

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the app
streamlit run nba_dashboard.py
```

## ⚠️ Notes

- NBA Stats API sometimes throttles or returns empty data—refresh or retry if needed.
- Pull requests and issues are welcome!
- Credit to JP Hwang's [blog post](https://www.jphwang.com/posts/interactive-basketball-data-visualizations-with-plotly/) for inspiration and plotly court logic!

