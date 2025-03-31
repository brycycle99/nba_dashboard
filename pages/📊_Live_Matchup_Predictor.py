# nba_win_predictor_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2
from nba_api.stats.static import teams
import plotly.graph_objects as go
from models.matchup_model_rf import train_model

if "model" not in st.session_state:
    with st.spinner("Training model on latest NBA data..."):
        model, acc, cv_mean, cv_std = train_model()
        st.session_state.model = model
        st.session_state.metrics = {
            "accuracy": acc,
            "cv_mean": cv_mean,
            "cv_std": cv_std
        }

st.title("NBA Win Probability Dashboard")

@st.cache_data
def get_teams():
    return pd.DataFrame(teams.get_teams())

team_df = get_teams()

@st.cache_data
def get_upcoming_games(date):
    scoreboard = ScoreboardV2(game_date=date.strftime('%m/%d/%Y'))
    return scoreboard.get_data_frames()[0]

@st.cache_data
def get_league_game_log():
    game_log = LeagueGameLog(season="2023-24", player_or_team_abbreviation="T")
    df = game_log.get_data_frames()[0]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    return df.sort_values('GAME_DATE')

league_df = get_league_game_log()

selected_date = st.date_input("Select Date", datetime.date.today())
date_timestamp = pd.Timestamp(selected_date)
games_today = get_upcoming_games(selected_date)
games_today = games_today[['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_DATE_EST']].dropna()

def get_team_name(team_id):
    row = team_df[team_df['id'] == int(team_id)]
    return row['full_name'].values[0] if not row.empty else "Unknown"

games_today['MATCHUP'] = games_today.apply(
    lambda x: f"{get_team_name(x['VISITOR_TEAM_ID'])} @ {get_team_name(x['HOME_TEAM_ID'])}", axis=1
)

selected_matchup = st.selectbox("Choose a Game", games_today['MATCHUP'].values)
row = games_today[games_today['MATCHUP'] == selected_matchup].iloc[0]
home_id, away_id = row['HOME_TEAM_ID'], row['VISITOR_TEAM_ID']

def get_recent_avg_from_league(team_id, ref_timestamp):
    df = league_df[(league_df['TEAM_ID'] == team_id) & (league_df['GAME_DATE'] < ref_timestamp)].copy()
    df = df.sort_values('GAME_DATE', ascending=False).head(15)
    df['WIN'] = df['WL'].map({'W': 1, 'L': 0})

    df['DAYS_SINCE_LAST_GAME'] = (df['GAME_DATE'] - df['GAME_DATE'].shift(-1)).dt.days
    df['IS_BACK_TO_BACK'] = (df['DAYS_SINCE_LAST_GAME'] == 1).astype(int)

    required_stats = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'AST', 'REB', 'TOV', 'PLUS_MINUS']
    available_stats = [s for s in required_stats if s in df.columns]

    for n in range(5, 2, -1):
        recent = df[available_stats].head(n)
        if recent.notna().sum().min() == recent.shape[0]:
            avg_stats = recent.mean()
            break
    else:
        avg_stats = pd.Series(dtype=float)

    try:
        elo = df['WIN'].rolling(10, min_periods=3).mean().iloc[-1]
    except IndexError:
        elo = np.nan

    rest_days = df['DAYS_SINCE_LAST_GAME'].iloc[0] if 'DAYS_SINCE_LAST_GAME' in df and not df['DAYS_SINCE_LAST_GAME'].isna().iloc[0] else np.nan
    is_b2b = df['IS_BACK_TO_BACK'].iloc[0] if 'IS_BACK_TO_BACK' in df else 0

    return avg_stats, elo, rest_days, is_b2b

home_stats, home_elo, home_rest, home_b2b = get_recent_avg_from_league(home_id, date_timestamp)
away_stats, away_elo, away_rest, away_b2b = get_recent_avg_from_league(away_id, date_timestamp)

def create_features(home, away, home_elo, away_elo, home_id, away_id, home_rest, away_rest, home_b2b, away_b2b):
    if home.empty or away.empty or pd.isna(home_elo) or pd.isna(away_elo):
        return pd.DataFrame()
    base_features = {
        'FG_PCT_DIFF': home.get('FG_PCT', 0) - away.get('FG_PCT', 0),
        'FG3_PCT_DIFF': home.get('FG3_PCT', 0) - away.get('FG3_PCT', 0),
        'FT_PCT_DIFF': home.get('FT_PCT', 0) - away.get('FT_PCT', 0),
        'AST_DIFF': home.get('AST', 0) - away.get('AST', 0),
        'REB_DIFF': home.get('REB', 0) - away.get('REB', 0),
        'TOV_DIFF': home.get('TOV', 0) - away.get('TOV', 0),
        'PLUS_MINUS_DIFF': home.get('PLUS_MINUS', 0) - away.get('PLUS_MINUS', 0),
        'ELO_DIFF': home_elo - away_elo,
        'REST_DIFF': home_rest - away_rest,
        'B2B_DIFF': home_b2b - away_b2b
    }
    home_onehot = {f'HT_{int(home_id)}': 1}
    away_onehot = {f'AT_{int(away_id)}': 1}
    full_row = {**base_features, **home_onehot, **away_onehot}

    model = st.session_state.model
    model_features = model.feature_names_in_
    full_df = pd.DataFrame([full_row])
    for col in model_features:
        if col not in full_df.columns:
            full_df[col] = 0
    return full_df[model_features]

features = create_features(home_stats, away_stats, home_elo, away_elo, home_id, away_id, home_rest, away_rest, home_b2b, away_b2b)
if features.empty:
    st.error("Feature generation failed. Possibly missing team stats or Elo values.")
    st.stop()

model = st.session_state.model
pred_proba = model.predict_proba(features)[0][1]

home_name = get_team_name(home_id)
away_name = get_team_name(away_id)

st.subheader("Win Probability")
st.markdown(
    "This chart displays the predicted win probability for the selected matchup. "
    "Predictions are generated using a Random Forest machine learning model trained on team statistics, "
    "recent performance, Elo ratings, rest days, and more."
)
# Label translations
stat_labels = {
    'FG_PCT': "Field Goal %",
    'FG3_PCT': "3-Point %",
    'FT_PCT': "Free Throw %",
    'AST': "Assists per game",
    'REB': "Rebounds per game",
    'TOV': "Turnovers per game",
    'PLUS_MINUS': "Average scoring margin",
    'ELO': "Recent Elo (win %)",
    'REST': "Days since last game",
    'B2B': "Played back-to-back?"
}

# Hover text formatter
def format_stats(stats, label, elo=None, rest=None, b2b=None):
    if stats.empty:
        return f"{label} stats not available"
    lines = [f"<b>{label} - Recent 5-game averages:</b>"]
    for stat, desc in stat_labels.items():
        val = (
            stats.get(stat, None)
            if stat in stats else
            elo if stat == "ELO" else
            rest if stat == "REST" else
            b2b if stat == "B2B" else None
        )
        if val is not None and not pd.isna(val):
            lines.append(f"{desc}: {val:.3f}")
    return "<br>".join(lines)

home_hover = format_stats(home_stats, "Home Team", elo=home_elo, rest=home_rest, b2b=home_b2b)
away_hover = format_stats(away_stats, "Away Team", elo=away_elo, rest=away_rest, b2b=away_b2b)

# Build chart
fig = go.Figure()
fig.add_trace(go.Bar(
    x=[pred_proba * 100],
    y=[home_name],
    orientation='h',
    name=home_name,
    hovertext=home_hover,
    hoverinfo='text'
))
fig.add_trace(go.Bar(
    x=[(1 - pred_proba) * 100],
    y=[away_name],
    orientation='h',
    name=away_name,
    hovertext=away_hover,
    hoverinfo='text'
))
fig.update_layout(barmode='stack', xaxis_title='Win Probability (%)')
st.plotly_chart(fig, use_container_width=True)

importances = model.feature_importances_
feature_names = features.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

st.subheader("📊 Model Evaluation")
st.markdown(
    "This section shows how well the model performs overall — including its accuracy on a test split "
    "and average cross-validation performance across multiple data folds."
)
metrics = st.session_state.get("metrics", {})
if metrics:
    st.markdown(f"> Training Accuracy: {metrics['accuracy']:.3f}")
    st.markdown(f"> Cross-Validation Accuracy: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
else:
    st.warning("Model metrics not available.")

st.subheader("🔍 Feature Importance")
st.markdown(
    "This table ranks which features had the most influence on the model’s prediction. "
    "Higher importance means the model relied more on that feature when estimating the outcome."
)
st.dataframe(importance_df)


