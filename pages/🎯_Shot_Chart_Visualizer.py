import streamlit as st
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import players, teams
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def draw_plotly_court(fig, fig_width=600, margins=10):

    import numpy as np

    # From: https://community.plot.ly/t/arc-shape-with-path/7205/5
    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200, closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    fig_height = fig_width * (470 + 2 * margins) / (500 + 2 * margins)
    fig.update_layout(width=fig_width, height=fig_height)

    fig.update_xaxes(range=[-250 - margins, 250 + margins])
    fig.update_yaxes(range=[-52.5 - margins, 417.5 + margins])

    threept_break_y = 89.47765084
    three_line_col = "#777777"
    main_line_col = "#777777"

    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        shapes=[
            dict(
                type="rect", x0=-250, y0=-52.5, x1=250, y1=417.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="rect", x0=-80, y0=-52.5, x1=80, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="rect", x0=-60, y0=-52.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),
            dict(
                type="circle", x0=-60, y0=77.5, x1=60, y1=197.5, xref="x", yref="y",
                line=dict(color=main_line_col, width=1),
                # fillcolor='#dddddd',
                layer='below'
            ),
            dict(
                type="line", x0=-60, y0=137.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                layer='below'
            ),

            dict(
                type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5,
                line=dict(color="#ec7607", width=1),
                fillcolor='#ec7607',
            ),
            dict(
                type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, xref="x", yref="y",
                line=dict(color="#ec7607", width=1),
            ),
            dict(
                type="line", x0=-30, y0=-12.5, x1=30, y1=-12.5,
                line=dict(color="#ec7607", width=1),
            ),

            dict(type="path",
                 path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(type="path",
                 path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101, end_angle=np.pi - 0.386283101),
                 line=dict(color=main_line_col, width=1), layer='below'),
            dict(
                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=220, y0=-52.5, x1=220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ),

            dict(
                type="line", x0=-250, y0=227.5, x1=-220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=250, y0=227.5, x1=220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=17.5, x1=-80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=27.5, x1=-80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=57.5, x1=-80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=-90, y0=87.5, x1=-80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=17.5, x1=80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=27.5, x1=80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=57.5, x1=80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),
            dict(
                type="line", x0=90, y0=87.5, x1=80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ),

            dict(type="path",
                 path=ellipse_arc(y_center=417.5, a=60, b=60, start_angle=-0, end_angle=-np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'),

        ]
    )
    return True


def plot_data(data):

    x = data['LOC_X'].values
    y = data['LOC_Y'].values

    gridsize = (30)

    hb_attempts = plt.hexbin(x, y, gridsize=gridsize, cmap='gray')
    coords = hb_attempts.get_offsets()
    plt.close()


    coords = hb_attempts.get_offsets()

    attempts = hb_attempts.get_array()

    mask = attempts >= 2
    coords = coords[mask]
    attempts = attempts[mask]
    total_attempts = attempts.sum()

    attempt_freq = np.nan_to_num((attempts / total_attempts))
    customdata = np.stack((attempts, attempt_freq), axis=1)
    colorscale = [
        [0.0, 'rgba(0, 0, 250, .00001)'], 
        [0.25, 'rgba(0, 0, 250, .1)'],
        [0.5, 'rgba(0, 0, 250, .5)'],  
        [1.0, 'rgba(0, 0, 250, 1.0)']   
    ]

    fig.add_trace(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode='markers',
            marker=dict(
                size=20,
                color=attempts,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title='Attempts',
                ),
                line=dict(width=0),
                sizemode='diameter',
                symbol='hexagon',
                cmin = 0,    
                cmax = safe_percentile(attempts, 90)
            ),
            hovertemplate=
                'Attempts: %{customdata[0]}<br>' +
                'Freq: %{customdata[1]:.1%}<br>',
            customdata=customdata
        )
    )


    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))
    
def safe_percentile(arr, q, default=None):
    try:
        return np.percentile(arr, q)
    except Exception:
        return default

def get_player_id(name):
    player_id = [player for player in players_dict if player['full_name']==name][0]['id']
    return player_id

def get_team_id(name):
    team_id = [team for team in teams_dict if team['full_name']==name][0]['id']
    return team_id

def req_ShotChartDetail(player_id, team_id, season_year, season_type):
    all_shots_dict = shotchartdetail.ShotChartDetail(team_id=team_id, player_id=player_id, season_nullable=season_year, season_type_all_star=season_type)
    all_shots_df = all_shots_dict.get_data_frames()[0]
    return all_shots_df



players_dict = players.get_players()

teams_dict = teams.get_teams()

player_list = [player['full_name'] for player in players_dict]

team_list = [team['full_name'] for team in teams_dict]

fig = go.Figure()

draw_plotly_court(fig)

st.title('Shot Chart')

seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(1996, 2025)]
season_year = st.selectbox('Select season:', seasons, placeholder='Choose an option', index=None)
season_type = st.selectbox('Select season type:', ['Regular Season', 'Playoffs'], placeholder='Choose an option', index=None)
mode_select = st.selectbox('Select viewing mode:', ['League-wide', 'Team', 'Player'], placeholder='Choose an option', index=None)

if (mode_select == 'League-wide') and season_year and season_type:
    all_shots_df = req_ShotChartDetail(0, 0, season_year, season_type)
    plot_data(all_shots_df)
    

if (mode_select == 'Team') and season_year and season_type:
    team_name = st.selectbox('Select team:', team_list, placeholder='Choose an option', index=None)
    if team_name:
        team_id = get_team_id(team_name)
        all_shots_df = req_ShotChartDetail(0, team_id, season_year, season_type)
        plot_data(all_shots_df)

if (mode_select == 'Player') and season_year and season_type:
    player_name = st.selectbox('Select player:', player_list, placeholder='Choose an option', index=None)
    if player_name:
        player_id = get_player_id(player_name)
        all_shots_df = req_ShotChartDetail(player_id, 0, season_year, season_type)
        plot_data(all_shots_df)






