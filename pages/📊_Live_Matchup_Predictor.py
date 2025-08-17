# app.py
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from nba_api.stats.endpoints import ScoreboardV2, LeagueGameLog
try:
    from nba_api.stats.endpoints import ScoreboardV3  # type: ignore
    HAS_V3 = True
except Exception:
    HAS_V3 = False

from nba_api.stats.static import teams
from models.matchup_model_lr import train_model, FEATURES, STAT_COLS, SEASONS

st.set_page_config(page_title="NBA Win Predictor — Logistic Regression", page_icon="🏀", layout="centered")
st.title("🏀 NBA Win Predictor — Logistic Regression")
st.caption("Features: last-5 shooting & possession diffs, **rolling win %** (last 10) diff, and **B2B** diff. Trained on multiple seasons.")

# -------------------- Season helpers --------------------
def season_from_date(d: datetime.date) -> str:
    year = d.year if d.month >= 10 else d.year - 1
    return f"{year}-{str((year + 1) % 100).zfill(2)}"

def season_back(season_label: str, k: int = 1) -> str:
    start = int(season_label[:4])
    back = start - k
    return f"{back}-{str((back + 1) % 100).zfill(2)}"

def seasons_for_ref(ref_ts: pd.Timestamp, max_lookback: int = 3) -> list[str]:
    s0 = season_from_date(ref_ts.date())
    out = [s0]
    for k in range(1, max_lookback + 1):
        out.append(season_back(s0, k))
    return out

# -------------------- Caching --------------------
@st.cache_resource(show_spinner=True)
def load_model_and_metrics():
    model, metrics = train_model(force_retrain=False, seasons=SEASONS)
    return model, metrics

@st.cache_data(show_spinner=False)
def get_teams_df():
    return pd.DataFrame(teams.get_teams())

@st.cache_data(show_spinner=False)
def get_league_logs_multi(seasons: list[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        df = LeagueGameLog(season=s, player_or_team_abbreviation="T").get_data_frames()[0].copy()
        df["SEASON"] = s
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df["WIN"] = df["WL"].map({"W": 1, "L": 0})
        frames.append(df.sort_values(["TEAM_ID", "GAME_DATE"]))
    return pd.concat(frames, ignore_index=True).sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def get_scoreboard(date: datetime.date) -> pd.DataFrame:
    # Prefer V3 when available
    if HAS_V3:
        try:
            sb3 = ScoreboardV3(game_date=date.strftime('%Y-%m-%d'))
            for df in sb3.get_data_frames() or []:
                if {'gameId','homeTeamId','awayTeamId'}.issubset(df.columns):
                    out = df[['gameId','homeTeamId','awayTeamId']].copy()
                    out.columns = ['GAME_ID','HOME_TEAM_ID','VISITOR_TEAM_ID']
                    out['GAME_DATE_EST'] = pd.Timestamp(date)
                    out['HOME_TEAM_ID'] = out['HOME_TEAM_ID'].astype(int)
                    out['VISITOR_TEAM_ID'] = out['VISITOR_TEAM_ID'].astype(int)
                    return out[['GAME_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','GAME_DATE_EST']]
        except Exception:
            pass
    # Fallback V2
    try:
        sb2 = ScoreboardV2(game_date=date.strftime('%m/%d/%Y'), league_id='00', day_offset=0)
        frames = sb2.get_data_frames() or []
        if frames:
            df0 = frames[0]
            keep = [c for c in ['GAME_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','GAME_DATE_EST'] if c in df0.columns]
            if keep:
                out = df0[keep].dropna().copy()
                if 'GAME_DATE_EST' not in out.columns:
                    out['GAME_DATE_EST'] = pd.Timestamp(date)
                out['HOME_TEAM_ID'] = out['HOME_TEAM_ID'].astype(int)
                out['VISITOR_TEAM_ID'] = out['VISITOR_TEAM_ID'].astype(int)
                return out[['GAME_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','GAME_DATE_EST']]
    except Exception:
        pass
    return pd.DataFrame(columns=['GAME_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','GAME_DATE_EST'])

# -------------------- Feature builders --------------------
TEAM_DF = get_teams_df()

def team_name(team_id: int) -> str:
    row = TEAM_DF[TEAM_DF['id'] == int(team_id)]
    return row['full_name'].values[0] if not row.empty else f"Team {team_id}"

def _recent_pre_game_vector_multi(league_df_multi: pd.DataFrame, team_id: int, ref_ts: pd.Timestamp,
                                  min_games: int = 3) -> dict:
    df = league_df_multi[(league_df_multi["TEAM_ID"] == team_id) &
                         (league_df_multi["GAME_DATE"] < ref_ts)].copy()
    if df.empty:
        return {}
    df = df.sort_values("GAME_DATE")
    if len(df) < min_games:
        return {}

    rolled = df.copy()
    rolled[STAT_COLS] = rolled[STAT_COLS].rolling(5, min_periods=min_games).mean()
    rolled["ROLL_WIN_PCT"] = rolled["WIN"].rolling(10, min_periods=min_games).mean()

    last = rolled.iloc[-1]
    needed = STAT_COLS + ["ROLL_WIN_PCT"]
    if pd.isna(last[needed]).any():
        return {}

    last_game_date = df.iloc[-1]["GAME_DATE"]
    b2b_current = int((ref_ts.normalize() - pd.Timestamp(last_game_date).normalize()).days == 1)

    return {
        "FG_PCT": float(last["FG_PCT"]),
        "FG3_PCT": float(last["FG3_PCT"]),
        "FT_PCT": float(last["FT_PCT"]),
        "REB": float(last["REB"]),
        "TOV": float(last["TOV"]),
        "ROLL_WIN_PCT": float(last["ROLL_WIN_PCT"]),
        "B2B": b2b_current,
    }

def build_matchup_features_multi(league_df_multi: pd.DataFrame, home_id: int, away_id: int,
                                 ref_ts: pd.Timestamp) -> pd.DataFrame:
    home = _recent_pre_game_vector_multi(league_df_multi, home_id, ref_ts, min_games=3)
    away = _recent_pre_game_vector_multi(league_df_multi, away_id, ref_ts, min_games=3)
    if not home or not away:
        return pd.DataFrame()

    row = {
        "FG_PCT_DIFF":       home["FG_PCT"] -        away["FG_PCT"],
        "FG3_PCT_DIFF":      home["FG3_PCT"] -       away["FG3_PCT"],
        "FT_PCT_DIFF":       home["FT_PCT"] -        away["FT_PCT"],
        "REB_DIFF":          home["REB"] -           away["REB"],
        "TOV_DIFF":          home["TOV"] -           away["TOV"],
        "ROLL_WIN_PCT_DIFF": home["ROLL_WIN_PCT"] -  away["ROLL_WIN_PCT"],
        "B2B_DIFF":          home["B2B"] -           away["B2B"],
    }
    return pd.DataFrame([row], columns=FEATURES).astype("float32")

def _feature_names_for(model) -> list:
    names = getattr(model, 'feature_names_in_', None)
    if names is None and hasattr(model.named_steps['lr'], 'feature_names_in_'):
        names = model.named_steps['lr'].feature_names_in_
    return list(names) if names is not None else list(FEATURES)

def predict_win_prob(model, X: pd.DataFrame) -> float:
    fnames = _feature_names_for(model)
    for col in fnames:
        if col not in X.columns:
            X[col] = 0
    X = X[fnames]
    proba = model.predict_proba(X)[0, 1]  # P(home win)
    return float(proba)

def get_lr_coefficients(model) -> pd.DataFrame:
    lr = model.named_steps['lr']
    coefs = lr.coef_.ravel()
    feats = _feature_names_for(model)
    df = pd.DataFrame({'Feature': feats, 'Coefficient': coefs})
    df['|Coefficient|'] = df['Coefficient'].abs()
    return df.sort_values('|Coefficient|', ascending=False).drop(columns='|Coefficient|')

# -------------------- UI --------------------
model, metrics = load_model_and_metrics()

col_left, col_right = st.columns([2, 1], vertical_alignment='center')
with col_left:
    st.subheader("Pick a date & game")
with col_right:
    st.caption(f"Trained seasons: {', '.join(SEASONS)}")

selected_date = st.date_input("Game date", datetime.date.today())
games_today = get_scoreboard(selected_date)

if games_today.empty:
    st.info("No games scheduled or schedule unavailable for this date.")
    st.stop()

TEAM_NAMES = TEAM_DF.set_index('id')['full_name'].to_dict()
games_today['HOME_NAME'] = games_today['HOME_TEAM_ID'].map(TEAM_NAMES).fillna(games_today['HOME_TEAM_ID'].astype(str))
games_today['AWAY_NAME'] = games_today['VISITOR_TEAM_ID'].map(TEAM_NAMES).fillna(games_today['VISITOR_TEAM_ID'].astype(str))
games_today['LABEL'] = games_today['AWAY_NAME'] + " @ " + games_today['HOME_NAME']

choice = st.selectbox("Choose a game", games_today['LABEL'].tolist())
row = games_today.loc[games_today['LABEL'] == choice].iloc[0]
home_id, away_id = int(row['HOME_TEAM_ID']), int(row['VISITOR_TEAM_ID'])
home_name, away_name = TEAM_NAMES.get(home_id, f"Team {home_id}"), TEAM_NAMES.get(away_id, f"Team {away_id}")

ref_ts = pd.Timestamp(selected_date)
season_list = seasons_for_ref(ref_ts, max_lookback=3)
LEAGUE_MULTI = get_league_logs_multi(season_list)

X = build_matchup_features_multi(LEAGUE_MULTI, home_id, away_id, ref_ts)
if X.empty:
    st.warning("Not enough prior games across seasons to build rolling stats for one or both teams.")
    st.stop()

p_home = predict_win_prob(model, X)

st.subheader("Win Probability")
fig = go.Figure()
fig.add_trace(go.Bar(x=[p_home * 100], y=[home_name], orientation='h'))
fig.add_trace(go.Bar(x=[(1 - p_home) * 100], y=[away_name], orientation='h'))
fig.update_layout(
    barmode='stack',
    xaxis_title='Win Probability (%)',
    showlegend=False,
    height=230,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Model Performance (training seasons)")
st.markdown(f"- **Time-split holdout accuracy:** {metrics['accuracy']:.3f}")
st.markdown(f"- **TimeSeries CV (3-fold):** {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")

with st.expander("Show logistic regression coefficients"):
    coef_df = get_lr_coefficients(model)
    st.dataframe(coef_df, use_container_width=True)

st.caption("Built from **home − away** pre-game rolling stats (last-5 games), **rolling win %** (last-10), and a **B2B** flag (diff).")
