# 🏀 NBA Shot Chart Visualizer

This Streamlit web application allows you to explore NBA shot chart data for different seasons. You can visualize shot attempt densities across the court for:

- The entire league  
- Specific NBA teams  
- Individual players  

Data is fetched using the [NBA Stats API](https://github.com/swar/nba_api), and visualized using [Plotly](https://plotly.com/python/), with a custom-drawn NBA half-court background.

---

## 🔧 Features

- 📅 **Season Selector**: Choose from NBA seasons 1996–2024  
- 🏆 **Season Type**: Regular Season or Playoffs  
- 🎯 **Modes**:
  - **League-wide**: See where shots are attempted most across the entire NBA
  - **Team**: View shot distributions by team
  - **Player**: Dive into individual player shot tendencies

- 🧊 **Hexbin Shot Map**: See shot attempt frequency using hexagonal binning with hover info  
- 🏟️ **Accurate Court Rendering**: Full half-court visual using Plotly shapes  

---

## 🧪 Requirements

-streamlit
-nba_api
-numpy
-plotly
-matplotlib

## 🚀 How to Run

### On Streamlit Cloud (Recommended)

Just click the **"Deploy to Streamlit"** button or upload this repo to [https://streamlit.io/cloud](https://streamlit.io/cloud). It will automatically detect `streamlit_app.py` and install dependencies from `requirements.txt`.

### Run Locally

```bash
# Step 1: Clone the repo
git clone https://github.com/your-username/nba-shot-chart.git
cd nba-shot-chart

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the app
streamlit run streamlit_app.py


