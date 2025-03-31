# model_training.py — refactored as reusable function

import pandas as pd
import numpy as np
import joblib
from nba_api.stats.endpoints import LeagueGameLog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from nba_api.stats.static import teams


def train_model():
    print("⏳ Downloading game logs...")
    game_logs = LeagueGameLog(
        season="2023-24",
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T"
    )
    games_df = game_logs.get_data_frames()[0]

    # Clean columns
    games_df = games_df[['TEAM_ID', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL',
                         'FG_PCT', 'FG3_PCT', 'FT_PCT', 'AST', 'REB', 'TOV', 'PLUS_MINUS']]
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
    games_df = games_df.sort_values('GAME_DATE')
    games_df['WIN'] = games_df['WL'].map({'W': 1, 'L': 0})

    # Rolling averages
    print("🔁 Computing 5-game averages...")
    rolling_stats = []
    stat_cols = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'AST', 'REB', 'TOV', 'PLUS_MINUS']
    for team_id in games_df['TEAM_ID'].unique():
        team_data = games_df[games_df['TEAM_ID'] == team_id].copy()
        team_data = team_data.sort_values('GAME_DATE')
        team_data[stat_cols] = team_data[stat_cols].rolling(5).mean()
        team_data['DAYS_SINCE_LAST_GAME'] = team_data['GAME_DATE'].diff().dt.days
        team_data['IS_BACK_TO_BACK'] = (team_data['DAYS_SINCE_LAST_GAME'] == 1).astype(int)
        rolling_stats.append(team_data)

    games_df = pd.concat(rolling_stats)
    games_df.dropna(inplace=True)

    # Elo ratings
    print("⚖️ Computing Elo-style win rates...")
    elo_df = games_df.groupby('TEAM_ID')['WIN'].rolling(10).mean().reset_index()
    elo_df.rename(columns={'WIN': 'ELO_SCORE'}, inplace=True)
    games_df = games_df.reset_index().merge(elo_df[['level_1', 'ELO_SCORE']], left_index=True, right_on='level_1')
    games_df.drop(columns=['index', 'level_1'], inplace=True)

    # Pair home/away games
    print("🔄 Pairing games...")
    paired_rows = []
    for game_id, group in games_df.groupby('GAME_ID'):
        if len(group) != 2:
            continue
        team1, team2 = group.iloc[0], group.iloc[1]

        if 'vs.' in team1['MATCHUP']:
            home, away = team1, team2
        else:
            home, away = team2, team1

        paired_rows.append({
            'FG_PCT_DIFF': home['FG_PCT'] - away['FG_PCT'],
            'FG3_PCT_DIFF': home['FG3_PCT'] - away['FG3_PCT'],
            'FT_PCT_DIFF': home['FT_PCT'] - away['FT_PCT'],
            'AST_DIFF': home['AST'] - away['AST'],
            'REB_DIFF': home['REB'] - away['REB'],
            'TOV_DIFF': home['TOV'] - away['TOV'],
            'PLUS_MINUS_DIFF': home['PLUS_MINUS'] - away['PLUS_MINUS'],
            'ELO_DIFF': home['ELO_SCORE'] - away['ELO_SCORE'],
            'REST_DIFF': home['DAYS_SINCE_LAST_GAME'] - away['DAYS_SINCE_LAST_GAME'],
            'B2B_DIFF': home['IS_BACK_TO_BACK'] - away['IS_BACK_TO_BACK'],
            'WIN': home['WIN'],
            'HOME_TEAM_ID': home['TEAM_ID'],
            'AWAY_TEAM_ID': away['TEAM_ID']
        })

    df = pd.DataFrame(paired_rows)
    df = pd.get_dummies(df, columns=['HOME_TEAM_ID', 'AWAY_TEAM_ID'], prefix=['HT', 'AT'])
    df.dropna(inplace=True)

    # Feature renaming
    X = df.drop(columns=['WIN'])
    team_map = {str(t['id']): t['full_name'] for t in teams.get_teams()}
    X = X.rename(columns={
        col: f"Home: {team_map[col.split('_')[1]]}" if col.startswith("HT_") else
             f"Away: {team_map[col.split('_')[1]]}" if col.startswith("AT_") else col
        for col in X.columns if col.startswith("HT_") or col.startswith("AT_")
    })
    y = df['WIN']

    # Train model
    print("🤖 Training Random Forest...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Accuracy + CV
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"✅ Training accuracy: {acc:.3f}")
    print(f"📊 Cross-val: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return model
