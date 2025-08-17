# models/matchup_model_lr.py
import os
import numpy as np
import pandas as pd
import joblib

from nba_api.stats.endpoints import LeagueGameLog

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -------------------- Config --------------------
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "matchup_model_lr_multi_rollwpct.pkl")

STAT_COLS = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'TOV']

# Compact features (no HOME_IND)
FEATURES = [
    'FG_PCT_DIFF',
    'FG3_PCT_DIFF',
    'FT_PCT_DIFF',
    'REB_DIFF',
    'TOV_DIFF',
    'ROLL_WIN_PCT_DIFF',
    'B2B_DIFF',
]

# -------------------- Data building --------------------
def _download_team_game_logs_multi(seasons):
    frames = []
    for s in seasons:
        gl = LeagueGameLog(season=s, player_or_team_abbreviation="T")
        df = gl.get_data_frames()[0].copy()
        df['SEASON'] = s
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df['WIN'] = df['WL'].map({'W': 1, 'L': 0}).astype(float)
        keep = ['SEASON', 'TEAM_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WIN'] + STAT_COLS
        frames.append(df[keep].sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)

def _add_pre_game_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby(['SEASON', 'TEAM_ID'])

    df[STAT_COLS] = grp[STAT_COLS].transform(lambda x: x.rolling(5, min_periods=3).mean().shift(1))
    df['ROLL_WIN_PCT'] = grp['WIN'].transform(lambda s: s.rolling(10, min_periods=3).mean().shift(1))

    df['DAYS_SINCE_LAST_GAME'] = grp['GAME_DATE'].diff().dt.days
    df['IS_BACK_TO_BACK'] = (df['DAYS_SINCE_LAST_GAME'] == 1).astype(int)

    need = STAT_COLS + ['ROLL_WIN_PCT', 'IS_BACK_TO_BACK']
    return df.dropna(subset=need).reset_index(drop=True)

def _pair_games_build_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per game (home perspective) with diff features + label.
    Dedupe per side to avoid duplicate GAME_IDs.
    """
    m_home = df['MATCHUP'].astype(str).str.contains(r'\bvs\.', na=False, regex=True)
    m_away = df['MATCHUP'].astype(str).str.contains('@', na=False)

    home = (df.loc[m_home]
              .sort_values(['GAME_ID', 'GAME_DATE', 'TEAM_ID'])
              .drop_duplicates(subset=['GAME_ID'], keep='last')
              .copy())
    away = (df.loc[m_away]
              .sort_values(['GAME_ID', 'GAME_DATE', 'TEAM_ID'])
              .drop_duplicates(subset=['GAME_ID'], keep='last')
              .copy())

    common_ids = np.intersect1d(home['GAME_ID'].values, away['GAME_ID'].values)
    home = home[home['GAME_ID'].isin(common_ids)]
    away = away[away['GAME_ID'].isin(common_ids)]

    pair = home.merge(away, on='GAME_ID', suffixes=('_H', '_A'), how='inner')
    pair = pair[pair['TEAM_ID_H'] != pair['TEAM_ID_A']]

    out = pd.DataFrame({
        'FG_PCT_DIFF':       pair['FG_PCT_H'] -       pair['FG_PCT_A'],
        'FG3_PCT_DIFF':      pair['FG3_PCT_H'] -      pair['FG3_PCT_A'],
        'FT_PCT_DIFF':       pair['FT_PCT_H'] -       pair['FT_PCT_A'],
        'REB_DIFF':          pair['REB_H'] -          pair['REB_A'],
        'TOV_DIFF':          pair['TOV_H'] -          pair['TOV_A'],
        'ROLL_WIN_PCT_DIFF': pair['ROLL_WIN_PCT_H'] - pair['ROLL_WIN_PCT_A'],
        'B2B_DIFF':          pair['IS_BACK_TO_BACK_H'] - pair['IS_BACK_TO_BACK_A'],
        'WIN':               pair['WIN_H'].astype(int),
        'GAME_DATE':         pair['GAME_DATE_H'],
    })

    cols = FEATURES + ['WIN', 'GAME_DATE']
    return out[cols].sort_values('GAME_DATE').reset_index(drop=True)

# -------------------- Model training & eval --------------------
def _train_lr(X: pd.DataFrame, y: pd.Series):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, solver='lbfgs'))
    ])
    pipe.fit(X, y)
    return pipe

def _time_based_holdout_eval(model, X: pd.DataFrame, y: pd.Series, dates: pd.Series, holdout_frac: float = 0.2) -> float:
    n = len(X)
    cut = int(n * (1 - holdout_frac))
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y.iloc[:cut], y.iloc[cut:]
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return float(accuracy_score(y_te, pred))

def _timeseries_cv_mean_std(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs = []
    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        accs.append(accuracy_score(y.iloc[te], pred))
    accs = np.array(accs, dtype=float)
    return float(accs.mean()), float(accs.std())

def train_model(force_retrain: bool = False, seasons=None, model_path: str = MODEL_PATH):
    seasons = seasons or SEASONS

    if (not force_retrain) and os.path.exists(model_path):
        model, metrics = joblib.load(model_path)
        return model, metrics

    raw = _download_team_game_logs_multi(seasons)
    feats = _add_pre_game_features(raw)
    df = _pair_games_build_diffs(feats)

    X = df[FEATURES].astype('float32')
    y = df['WIN'].astype(int)

    model = _train_lr(X, y)
    holdout_acc = _time_based_holdout_eval(model, X, y, df['GAME_DATE'], holdout_frac=0.2)
    cv_mean, cv_std = _timeseries_cv_mean_std(model, X, y, n_splits=3)

    model.fit(X, y)
    metrics = {'accuracy': holdout_acc, 'cv_mean': cv_mean, 'cv_std': cv_std}

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump((model, metrics), model_path)
    return model, metrics
