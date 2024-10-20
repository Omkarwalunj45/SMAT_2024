import streamlit as st
import pandas as pd
import math as mt
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title='Syed Mushtaq Ali Trophy Performance Analysis Portal', layout='wide')
st.title('Syed Mushtaq Ali Trophy Performance Analysis Portal')
csv_files = [
    "Dataset/SMAT2.csv",
    "Dataset/SMAT1_part1.csv",
    "Dataset/SMAT1_part2.csv"
]
dataframes = [pd.read_csv(csv_file, low_memory=False) for csv_file in csv_files]
pdf = pd.concat(dataframes, ignore_index=True)
cols_conv=['season','match_id']
pdf[cols_conv] = pdf[cols_conv].astype(str)
pdf=pdf.rename(columns={'innings':'inning'})
bpdf=pdf
idf = pd.read_csv("Dataset/lifesaver_bat_smat.csv",low_memory=False)

info_df=pd.read_csv("Dataset/cricket_players_data.csv",low_memory=False)
bidf=pd.read_csv("Dataset/lifesaver_bowl_smat.csv",low_memory=False)
info_df=info_df.rename(columns={'player':'Player_name'})
pdf[['noballs', 'wides','byes','legbyes','penalty']] = pdf[['noballs', 'wides','byes','legbyes','penalty']].fillna(0).astype(int)
pdf['valid_ball'] = pdf.apply(lambda x: 1 if (x['wides'] == 0 and x['noballs'] == 0) else 0, axis=1)
idf=idf[idf['final_year']=='2023/24']
def show_match_details(match_id):
    print("Hello")
    match_id = str(match_id)
    # Filter match details for the selected match_id
    match_details = pdf[pdf['match_id'] == match_id]
    print(match_details.head())
    print("DHENDNHEHCBDHBEDHBD")
    # First, remove duplicates based on match_id and ball within the same match
    print(f"Before removing duplicates based on 'match_id' and 'ball': {match_details.shape}")
    match_details = match_details.drop_duplicates(subset=['match_id', 'ball', 'inning','batsman','bowler','over'], keep='first')
    print(f"After removing duplicates based on 'match_id' and 'ball': {match_details.shape}")
    print("Hello")
    
    if not match_details.empty:
        st.write(f"### Match Details - Match ID: {match_id}")
        # Split the data by innings
        innings_1 = match_details[match_details['inning'] == 1]
        innings_2 = match_details[match_details['inning'] == 2]

        # Get batting teams for both innings
        batting_team_1 = innings_1['batting_team'].unique()[0] if not innings_1.empty else "Unknown"
        batting_team_2 = innings_2['batting_team'].unique()[0] if not innings_2.empty else "Unknown"

        # Show the scorecard for each innings
        if not innings_1.empty:
            total_runs_1 = innings_1['total_runs'].sum()
            total_balls_1 = (innings_1['valid_ball'].sum())%6
            total_overs_1=innings_1['over'].iloc[-1]
            # Handle special case for exactly 20 overs
            if (total_balls_1==0):
                overs_display = f"{total_overs_1}.{total_balls_1}"  # +1 because overs start from 1
            else:            
                overs_display = f"{total_overs_1-1}.{total_balls_1}"  # +1 because overs start from 1
            # Display innings result
            st.markdown(f"<h5 style='font-size: 30px;'>{innings_1['batting_team'].iloc[0]} Innings: {total_runs_1}/{innings_1['is_wkt'].sum()} ({overs_display} ov)</h5>", unsafe_allow_html=True)


            show_innings_scorecard(innings_1, f"Innings 1: {batting_team_1} Women")
        if not innings_2.empty:
            total_runs_2 = innings_2['total_runs'].sum()
            total_balls_2 = (innings_2['valid_ball'].sum())%6
            total_overs_2=innings_2['over'].iloc[-1]
            if (total_balls_2==0):
                overs_display = f"{total_overs_2}.{total_balls_2}"  # +1 because overs start from 1
            else:            
                overs_display = f"{total_overs_2-1}.{total_balls_2}"  # +1 because overs start from 1
            # Display innings result
            st.markdown(f"<h5 style='font-size: 30px;'>{innings_2['batting_team'].iloc[0]} Innings: {total_runs_2}/{innings_2['is_wkt'].sum()} ({overs_display} ov)</h5>", unsafe_allow_html=True)



            show_innings_scorecard(innings_2, f"Innings 2: {batting_team_2} Women")
    else:
        st.write("No match details found.")

def show_innings_scorecard(inning_data, title):
    # Batting scorecard
    st.write("Batting")
    batting_order = []
    
    # Iterate through the innings data to establish the batting order
    for i, row in inning_data.iterrows():
        batsman = row['batsman']
        non_striker = row['non_striker']
        
        if batsman not in batting_order:
            batting_order.append(batsman)
        if non_striker not in batting_order:
            batting_order.append(non_striker)
    
    # Calculate total extras
    total_extras = inning_data['extras'].sum()
    
    # Aggregate batting data for runs, balls faced, fours, and sixes
    batting_data = inning_data.groupby(['batsman']).agg({
        'batsman_runs': 'sum',
        'valid_ball': 'sum',
        'is_four': 'sum',
        'is_six': 'sum'
    }).reset_index()
    
    # Initialize columns for Wicket and Dismissal Kind
    batting_data['Wicket'] = "Not Out"  # Default value if no dismissal
    batting_data['Dismissal Kind'] = "-"  # Default value if no dismissal
    
    # Populate Wicket and Dismissal Kind based on dismissal events
    for index, row in batting_data.iterrows():
        batsman = row['batsman']
        
        # Check if this batsman was dismissed
        dismissed_data = inning_data[(inning_data['batsman'] == batsman) & (inning_data['is_wkt'] == 1)]
        
        if not dismissed_data.empty:
            dismissal_event = dismissed_data.iloc[0]
            
            # If bowler_wkt is 1, the bowler took the wicket
            if dismissal_event['bowler_wkt'] == 1:
                batting_data.at[index, 'Wicket'] = dismissal_event['bowler']
            else:
                batting_data.at[index, 'Wicket'] = "-"
            
            # Update dismissal kind
            batting_data.at[index, 'Dismissal Kind'] = dismissal_event['dismissal_kind']
        
        # Handle retired cases
        retired_data = inning_data[(inning_data['batsman'] == batsman) & (inning_data['dismissal_kind'] == 'retired')]
        if not retired_data.empty:
            retired_event = retired_data.iloc[-1]
            batting_data.at[index, 'Wicket'] = "-"
            batting_data.at[index, 'Dismissal Kind'] = retired_event['dismissal_kind']
    
    # Now handle players who are dismissed but have no valid balls faced
    for player in inning_data['player_dismissed'].unique():
        # Check if player is already in the batting_data
        if player not in batting_data['batsman'].values:
            # Get data for the dismissed player
            player_data = (inning_data[inning_data['player_dismissed'] == player])
            p_data = inning_data[inning_data['batsman'] == player]
            valid_ball_sum = p_data['valid_ball'].sum()         
            
            # Handling the case where the player is dismissed without facing a legal ball
            if valid_ball_sum == 0:
                dismissal_event = inning_data[inning_data['player_dismissed'] == player]  # Get the first row since it's a single dismissal event
                # dismissal_event = inning_data[inning_data['player_dismissed'] == player].iloc[0]
                if not dismissal_event.empty:
                    # bowler_wkt = dismissal_event['bowler_wkt'] #if isinstance(dismissal_event['bowler_wkt'], pd.Series) else dismissal_event['bowler_wkt']
                    
                    # Create a new row for the player to be added to the batting data
                    new_row = pd.DataFrame({
                        'batsman': [player],
                        'batsman_runs': [0],
                        'valid_ball': [0],
                        'is_four': [0],
                        'is_six': [0],
                        # 'Wicket': [dismissal_event['bowler'] if dismissal_event['bowler_wkt'] == 1 else '-'],
                        # 'Dismissal Kind': [dismissal_event['dismissal_kind']]
                        'Wicket': ["-"],  # Default value for Wicket
                        'Dismissal Kind': ["-"]
                    })
                    # Check if 'bowler_wkt' exists in the dismissal event data
                    new_row.at[0, 'Dismissal Kind'] = dismissal_event['dismissal_kind'].values[0] if isinstance(dismissal_event['dismissal_kind'], pd.Series) else dismissal_event['dismissal_kind']
                # Use pd.concat to add the new row to the existing DataFrame
                    batting_data = pd.concat([batting_data, new_row], ignore_index=True)
    
    # Calculate strike rate
    batting_data['batter_sr'] = (batting_data['batsman_runs'] / batting_data['valid_ball']).replace({0 : 0}) * 100
    
    # Rename columns for the batting scorecard
    batting_data.columns = ['Batsman', 'R', 'B', '4s', '6s', 'Wicket', 'Dismissal Kind', 'SR']
    
    # Filter out batsmen with 0 runs (if needed)
    batting_data['order'] = batting_data['Batsman'].apply(lambda x: batting_order.index(x) if x in batting_order else -1)
    batting_data = batting_data.sort_values(by='order').drop(columns='order').reset_index(drop=True)
    batting_data.index = batting_data.index + 1
    
    # Display the batting table
    st.table(batting_data)
    
    # Show extras
    st.write(f"**Extras:** {total_extras}")


    
    # Bowling scorecard
    st.write("Bowling")
    bowling_order = []
    
    for i, row in inning_data.iterrows():
        bowler = row['bowler']
        
        if bowler not in bowling_order:
            bowling_order.append(bowler)
    
    inning_data['adjusted_runs'] = inning_data.apply(lambda row: row['total_runs'] - (row['byes'] + row['legbyes'] + row['penalty']), axis=1)
    bowling_data = inning_data.groupby(['bowler']).agg({
        'valid_ball': 'sum',
        'adjusted_runs': 'sum',
        'bowler_wkt': 'sum',
        'wides': 'sum',
        'noballs': 'sum'
    }).reset_index()
    
    # Calculate overs bowled (converting balls to overs)
    bowling_data['Overs'] = (bowling_data['valid_ball'] // 6).astype(str) + "." + (bowling_data['valid_ball'] % 6).astype(str)
    
    # Calculate economy rate (total runs / overs)
    bowling_data['econ'] = bowling_data['adjusted_runs'] / (bowling_data['valid_ball'] / 6)
    
    # Calculate bowling strike rate (balls per wicket, avoid division by zero)
    bowling_data['bowl_sr'] = bowling_data['valid_ball'] / bowling_data['bowler_wkt']
    bowling_data['bowl_sr'] = bowling_data['bowl_sr'].replace([float('inf'), float('nan')], 0)
    
    # Select and rename columns for the bowling scorecard
    bowling_data = bowling_data[['bowler', 'Overs', 'adjusted_runs', 'bowler_wkt', 'wides', 'noballs', 'econ', 'bowl_sr']]
    bowling_data.columns = ['Bowler', 'O', 'R', 'W', 'WD', 'NB', 'Econ', 'SR']
    bowling_data = bowling_data[(bowling_data.Bowler) != '0']
    
    # Display bowling scorecard
    bowling_data['order'] =bowling_data['Bowler'].apply(lambda x: bowling_order.index(x))
    bowling_data = bowling_data.sort_values(by='order').drop(columns='order').reset_index(drop=True)
    bowling_data.index = bowling_data.index + 1
    st.table(bowling_data)


def categorize_phase(over):
              if over <= 6:
                  return 'Powerplay'
              elif 6 < over < 16:
                  return 'Middle'
              else:
                  return 'Death'
pdf['phase'] = pdf['over'].apply(categorize_phase)
def is_bowlers_wkt(player_dismissed,dismissal_kind):
  if type(player_dismissed)== str :
    if dismissal_kind not in ['run out','retired hurt','obstructing the field']:
      return 1
    else :
      return 0
  else:
    return 0
bpdf['bowler_wkt']=bpdf.apply(lambda x: (is_bowlers_wkt(x['player_dismissed'],x['dismissal_kind'])),axis=1)
# def round_up_floats(df, decimal_places=2):
#     # Round up only for float columns
#     float_cols = df.select_dtypes(include=['float'])
#     df[float_cols.columns] = np.ceil(float_cols * (10 ** decimal_places)) / (10 ** decimal_places)
#     return df
def round_up_floats(df, decimal_places=2):
    # Select only float columns from the DataFrame
    float_cols = df.select_dtypes(include=['float64', 'float32'])  # Ensure to catch all float types
    
    # Round up the float columns and maintain the same shape
    rounded_floats = np.ceil(float_cols * (10 ** decimal_places)) / (10 ** decimal_places)
    
    # Assign the rounded values back to the original DataFrame
    df[float_cols.columns] = rounded_floats
    
    return df

def standardize_season(season):
    if '/' in season:  # Check if the season is in 'YYYY/YY' format
          year = season.split('/')[0]  # Get the first part
    else:
          year = season  # Use as is if already in 'YYYY' format
    return year.strip()  # Return the year stripped of whitespace
def get_current_form(bpdf, player_name):
    # Filter for matches where the player batted or bowled
    player_matches = bpdf[(bpdf['batsman'] == player_name) | (bpdf['bowler'] == player_name)]
    player_matches['start_date'] = pd.to_datetime(player_matches['start_date'], format='%m/%d/%Y')
    player_matches = player_matches.sort_values(by='start_date', ascending=False)
    bpdf['start_date'] = pd.to_datetime(bpdf['start_date'], format='%m/%d/%Y')
    
    # Get the last 10 unique match IDs
    last_10_matches = player_matches['start_date'].drop_duplicates().sort_values(ascending=False).head(10)

    # Prepare the result DataFrame
    results = []

    for date in last_10_matches:
        # Get batting stats for this match
        bat_match_data = bpdf[(bpdf['start_date'] == date) & (bpdf['batsman'] == player_name)]
        match_id = None
        venue = None
        opp = None
        fan_pts_bat = 0
        fan_pts_bowl = 0
        
        if not bat_match_data.empty:
            runs = bat_match_data['batsman_runs'].sum() 
            balls_faced = bat_match_data['ball'].count()  # Sum balls faced
            SR = (runs / balls_faced) * 100 if balls_faced > 0 else 0.0
            venue = bat_match_data['venue'].iloc[0]
            match_id = bat_match_data['match_id'].iloc[0]
            date = bat_match_data['start_date'].iloc[0]
            opp = bat_match_data['bowling_team'].iloc[0]
        else:
            runs = 0
            balls_faced = 0
            SR = 0.0
        
        # Get bowling stats for this match
        bowl_match_data = bpdf[(bpdf['start_date'] == date) & (bpdf['bowler'] == player_name)]
        
        if not bowl_match_data.empty:
            balls_bowled = bowl_match_data['ball'].count()  # Sum balls bowled
            runs_given = bowl_match_data['total_runs'].sum()  # Sum runs given
            wickets = bowl_match_data['bowler_wkt'].sum()  # Sum wickets taken
            econ = (runs_given / (balls_bowled / 6)) if balls_bowled > 0 else 0.0  # Calculate Econ
            venue = bowl_match_data['venue'].iloc[0]
            match_id = bowl_match_data['match_id'].iloc[0]
            date = bowl_match_data['start_date'].iloc[0]
            opp = bowl_match_data['batting_team'].iloc[0]
        else:
            balls_bowled = 0
            runs_given = 0
            wickets = 0
            econ = 0.0
            
        results.append({
            "Date": date,
            "Match ID": match_id,
            "Runs": runs,
            "Balls Faced": balls_faced,
            "SR": SR,
            "Balls Bowled": balls_bowled,
            "Runs Given": runs_given,
            "Wickets": wickets,
            "Econ": econ,
            "Venue": venue,
            "Opponent": opp,
        })
    
    return pd.DataFrame(results)

def cumulator(temp_df):
    # First, remove duplicates based on match_id and ball within the same match
    print(f"Before removing duplicates based on 'match_id' and 'ball': {temp_df.shape}")
    temp_df = temp_df.drop_duplicates(subset=['match_id', 'ball','inning'], keep='first')
    print(f"After removing duplicates based on 'match_id' and 'ball': {temp_df.shape}")

    # Ensure 'total_runs' exists
    if 'total_runs' not in temp_df.columns:
        raise KeyError("Column 'total_runs' does not exist in temp_df.")

    # Calculate runs, balls faced, innings, dismissals, etc.
    runs = temp_df.groupby(['batsman'])['batsman_runs'].sum().reset_index().rename(columns={'batsman_runs': 'runs'})
    balls = temp_df.groupby(['batsman'])['ball'].count().reset_index()
    inn = temp_df.groupby(['batsman'])['match_id'].apply(lambda x: len(list(np.unique(x)))).reset_index().rename(columns={'match_id': 'innings'})
    matches = temp_df.groupby(['batsman'])['match_id'].nunique().reset_index().rename(columns={'match_id': 'matches'})
    dis = temp_df.groupby(['batsman'])['player_dismissed'].count().reset_index().rename(columns={'player_dismissed': 'dismissals'})
    sixes = temp_df.groupby(['batsman'])['is_six'].sum().reset_index().rename(columns={'is_six': 'sixes'})
    fours = temp_df.groupby(['batsman'])['is_four'].sum().reset_index().rename(columns={'is_four': 'fours'})
    dots = temp_df.groupby(['batsman'])['is_dot'].sum().reset_index().rename(columns={'is_dot': 'dots'})
    ones = temp_df.groupby(['batsman'])['is_one'].sum().reset_index().rename(columns={'is_one': 'ones'})
    twos = temp_df.groupby(['batsman'])['is_two'].sum().reset_index().rename(columns={'is_two': 'twos'})
    threes = temp_df.groupby(['batsman'])['is_three'].sum().reset_index().rename(columns={'is_three': 'threes'})
    bat_team = temp_df.groupby(['batsman'])['batting_team'].unique().reset_index()

    # Convert the array of countries to a string without brackets
    bat_team['batting_team'] = bat_team['batting_team'].apply(lambda x: ', '.join(x)).str.replace('[', '').str.replace(']', '')

    match_runs = temp_df.groupby(['batsman', 'match_id'])['batsman_runs'].sum().reset_index()

    # Count 100s, 50s, and 30s
    hundreds = match_runs[match_runs['batsman_runs'] >= 100].groupby('batsman').size().reset_index(name='hundreds')
    fifties = match_runs[(match_runs['batsman_runs'] >= 50) & (match_runs['batsman_runs'] < 100)].groupby('batsman').size().reset_index(name='fifties')
    thirties = match_runs[(match_runs['batsman_runs'] >= 30) & (match_runs['batsman_runs'] < 50)].groupby('batsman').size().reset_index(name='thirties')

    # Calculate the highest score for each batsman
    highest_scores = match_runs.groupby('batsman')['batsman_runs'].max().reset_index().rename(columns={'batsman_runs': 'highest_score'})

    # Merge all the calculated metrics into a single DataFrame
    summary_df = runs.merge(balls, on='batsman', how='left')
    summary_df = summary_df.merge(inn, on='batsman', how='left')
    summary_df = summary_df.merge(matches, on='batsman', how='left')
    summary_df = summary_df.merge(dis, on='batsman', how='left')
    summary_df = summary_df.merge(sixes, on='batsman', how='left')
    summary_df = summary_df.merge(fours, on='batsman', how='left')
    summary_df = summary_df.merge(dots, on='batsman', how='left')
    summary_df = summary_df.merge(ones, on='batsman', how='left')
    summary_df = summary_df.merge(twos, on='batsman', how='left')
    summary_df = summary_df.merge(threes, on='batsman', how='left')
    summary_df = summary_df.merge(bat_team, on='batsman', how='left')
    summary_df = summary_df.merge(hundreds, on='batsman', how='left')
    summary_df = summary_df.merge(fifties, on='batsman', how='left')
    summary_df = summary_df.merge(thirties, on='batsman', how='left')
    summary_df = summary_df.merge(highest_scores, on='batsman', how='left')

    # Calculating additional columns
    def bpd(balls, dis):
        return balls if dis == 0 else balls / dis
    
    def bpb(balls, bdry):
        return balls if bdry == 0 else balls / bdry
    
    def avg(runs, dis, inn):
        return runs / inn if dis == 0 else runs / dis
    
    def DP(balls, dots):
        return (dots / balls) * 100
    
    summary_df['SR'] = summary_df.apply(lambda x: (x['runs'] / x['ball']) * 100, axis=1)
    
    summary_df['BPD'] = summary_df.apply(lambda x: bpd(x['ball'], x['dismissals']), axis=1)
    summary_df['BPB'] = summary_df.apply(lambda x: bpb(x['ball'], (x['fours'] + x['sixes'])), axis=1)
    summary_df['nbdry_sr'] = summary_df.apply(lambda x: ((x['dots'] * 0 + x['ones'] * 1 + x['twos'] * 2 + x['threes'] * 3) /(x['dots'] + x['ones'] + x['twos'] + x['threes']) * 100) if (x['dots'] + x['ones'] + x['twos'] + x['threes']) > 0 else 0, axis=1)
    summary_df['AVG'] = summary_df.apply(lambda x: avg(x['runs'], x['dismissals'], x['innings']), axis=1)
    summary_df['dot_percentage'] = (summary_df['dots'] / summary_df['ball']) * 100

    debut_year = temp_df.groupby('batsman')['season'].min().reset_index()
    final_year = temp_df.groupby('batsman')['season'].max().reset_index()
    debut_year.rename(columns={'season': 'debut_year'}, inplace=True)
    final_year.rename(columns={'season': 'final_year'}, inplace=True)
    summary_df = summary_df.merge(debut_year, on='batsman').merge(final_year, on='batsman')

    return summary_df
def bcum(df):
    # First, remove duplicates based on match_id and ball within the same match
    print(f"Before removing duplicates based on 'match_id' and 'ball': {df.shape}")
    df = df.drop_duplicates(subset=['match_id', 'ball','inning'], keep='first')
    print(f"After removing duplicates based on 'match_id' and 'ball': {df.shape}")

    # Create various aggregates
    runs = pd.DataFrame(df.groupby(['bowler'])['batsman_runs'].sum()).reset_index().rename(columns={'batsman_runs': 'runs'})
    innings = pd.DataFrame(df.groupby(['bowler'])['match_id'].nunique()).reset_index().rename(columns={'match_id': 'innings'})
    balls = pd.DataFrame(df.groupby(['bowler'])['ball'].count()).reset_index().rename(columns={'ball': 'balls'})
    wkts = pd.DataFrame(df.groupby(['bowler'])['bowler_wkt'].sum()).reset_index().rename(columns={'bowler_wkt': 'wkts'})
    dots = pd.DataFrame(df.groupby(['bowler'])['is_dot'].sum()).reset_index().rename(columns={'is_dot': 'dots'})
    ones = pd.DataFrame(df.groupby(['bowler'])['is_one'].sum()).reset_index().rename(columns={'is_one': 'ones'})
    twos = pd.DataFrame(df.groupby(['bowler'])['is_two'].sum()).reset_index().rename(columns={'is_two': 'twos'})
    threes = pd.DataFrame(df.groupby(['bowler'])['is_three'].sum()).reset_index().rename(columns={'is_three': 'threes'})
    fours = pd.DataFrame(df.groupby(['bowler'])['is_four'].sum()).reset_index().rename(columns={'is_four': 'fours'})
    sixes = pd.DataFrame(df.groupby(['bowler'])['is_six'].sum()).reset_index().rename(columns={'is_six': 'sixes'})

    dismissals_count = df.groupby(['bowler', 'match_id'])['bowler_wkt'].sum()
    three_wicket_hauls = dismissals_count[dismissals_count >= 3].groupby('bowler').count().reset_index().rename(columns={'bowler_wkt': 'three_wicket_hauls'})
    bbi = dismissals_count.groupby('bowler').max().reset_index().rename(columns={'bowler_wkt': 'bbi'})

    # Identify maiden overs (group by match and over, check if total_runs == 0)
    df['over'] = df['ball'].apply(lambda x: int(x))  # Assuming ball represents the ball within an over
    maiden_overs = df.groupby(['bowler', 'match_id', 'over']).filter(lambda x: x['total_runs'].sum() == 0)
    maiden_overs_count = maiden_overs.groupby('bowler')['over'].count().reset_index().rename(columns={'over': 'maiden_overs'})

    # Merge all metrics into a single DataFrame
    bowl_rec = pd.merge(innings, balls, on='bowler')\
                 .merge(runs, on='bowler')\
                 .merge(wkts, on='bowler')\
                 .merge(sixes, on='bowler')\
                 .merge(fours, on='bowler')\
                 .merge(dots, on='bowler')\
                 .merge(three_wicket_hauls, on='bowler', how='left')\
                 .merge(maiden_overs_count, on='bowler', how='left')\
                 .merge(bbi, on='bowler', how='left')

    # Fill NaN values for bowlers with no 3W hauls or maiden overs
    bowl_rec['three_wicket_hauls'] = bowl_rec['three_wicket_hauls'].fillna(0)
    bowl_rec['maiden_overs'] = bowl_rec['maiden_overs'].fillna(0)
    debut_year = df.groupby('bowler')['season'].min().reset_index()
    final_year = df.groupby('bowler')['season'].max().reset_index()
    debut_year.rename(columns={'season': 'debut_year'}, inplace=True)
    final_year.rename(columns={'season': 'final_year'}, inplace=True)
    bowl_rec = bowl_rec.merge(debut_year, on='bowler').merge(final_year, on='bowler')

    # Calculate additional metrics
    bowl_rec['dot%'] = (bowl_rec['dots'] / bowl_rec['balls']) * 100

    # Check for zeros before performing calculations
    bowl_rec['avg'] = bowl_rec['runs'] / bowl_rec['wkts'].replace(0, np.nan)
    bowl_rec['sr'] = bowl_rec['balls'] / bowl_rec['wkts'].replace(0, np.nan)
    bowl_rec['econ'] = (bowl_rec['runs'] * 6 / bowl_rec['balls'].replace(0, np.nan))

    return bowl_rec
venue_state_map = {
    'Saurashtra Cricket Association Stadium': 'Gujarat',
    'Shaheed Veer Narayan Singh International Stadium': 'Chhattisgarh',
    'Arun Jaitley Stadium': 'Delhi',
    'Dr. Y.S. Rajasekhara Reddy ACA VDCA Cricket Stadium': 'Andhra Pradesh',
    'JSCA International Stadium Complex': 'Jharkhand',
    'Dr P.V.G. Raju ACA Sports Complex': 'Andhra Pradesh',
    'JU Second Campus, Salt Lake': 'West Bengal',
    'Eden Gardens': 'West Bengal',
    'Dr. Gokaraju Laila Ganga Raju ACA Cricket Complex -DVR Ground, Mulapadu': 'Andhra Pradesh',
    'Dr. Gokaraju Laila Ganga Raju ACA Cricket Complex -CP Ground, Mulapadu': 'Andhra Pradesh',
    'Lalbhai Contractor Stadium': 'Gujarat',
    'C B Patel Ground': 'Gujarat',
    'Holkar Stadium': 'Madhya Pradesh',
    'Emerald Heights International School Ground': 'Madhya Pradesh',
    'Barabati Stadium': 'Odisha',
    'DRIEMS Ground': 'Odisha',
    'Airforce Complex ground, Palam': 'Delhi',
    'Airforce Complex ground, Palam II': 'Delhi',
    "St'Xavier's KCA Cricket Ground": 'Kerala',
    'Greenfield Stadium': 'Kerala',
    'Cricket Stadium, Sector-16': 'Chandigarh',
    'GSSS, Sector 26': 'Chandigarh',
    'BKC Ground': 'Maharashtra',
    'Wankhede Stadium': 'Maharashtra',
    'Alur Cricket Stadium': 'Karnataka',
    'Alur Cricket Stadium II': 'Karnataka',
    'Alur Cricket Stadium III': 'Karnataka',
    'Jadavpur University Campus': 'West Bengal',
    'Motibaug Cricket Ground': 'Madhya Pradesh',
    'F B Colony Ground': 'Madhya Pradesh',
    'Reliance Cricket Stadium': 'Maharashtra',
    'Sharad Pawar Cricket Academy BKC': 'Maharashtra',
    'SSN College Ground': 'Tamil Nadu',
    'T I Murugappa Ground': 'Tamil Nadu',
    'Sri Ramachandra Medical College': 'Tamil Nadu',
    'IC-Gurunanak College Ground': 'Tamil Nadu',
    "Narendra Modi Stadium Ground 'A', Motera": 'Gujarat',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium': 'Uttar Pradesh',
    'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium B': 'Uttar Pradesh',
    'ACA Stadium, Barsapara': 'Assam',
    'Nehru Stadium': 'Delhi',
    'Gurugram Cricket Ground (SRNCC)': 'Haryana',
    'Chaudhry Bansi Lal Cricket Stadium': 'Haryana',
    'Gokaraju Liala Gangaaraju ACA Cricket Ground': 'Andhra Pradesh',
    'ACA Stadium, Mangalagiri': 'Andhra Pradesh',
    'Alembic 2 Cricket Ground': 'Gujarat',
    'Sawai Mansingh Stadium, Jaipur': 'Rajasthan',
    'Saurashtra Cricket Association Stadium, Rajkot': 'Gujarat',
    'Holkar Cricket Stadium, Indore': 'Madhya Pradesh',
    'Eden Gardens, Kolkata': 'West Bengal',
    'Jadavpur University Campus 2nd Ground, Kolkata': 'West Bengal',
    'JSCA International Stadium Complex, Ranchi': 'Jharkhand',
    'Abhimanyu Cricket Academy, Dehradun': 'Uttarakhand',
    'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab',
    'Vidarbha Cricket Association Stadium, Jamtha': 'Maharashtra',
    'VCA Ground': 'Maharashtra',
    'Jawaharlal Nehru Stadium': 'Uttar Pradesh',
    'St Pauls college ground Kalamassery': 'Kerala',
    'Alembic 1 Cricket Ground': 'Gujarat'
}
# Preprocess the debut column to extract the year
idf['debut_year'] = idf['debut_year'].str.split('/').str[0]  # Extract the year from "YYYY/YY"
pdf.rename(columns={'batting Style': 'batting_style','bowling Style': 'bowling_style'}, inplace=True)
bowling_style_mapping = {
    'Righ-arm medium fast ': 'Right-arm medium fast',
    'Right arm Medium fast': 'Right-arm medium fast',
    'Right-arm Medium fast': 'Right-arm medium fast',
    'Right-arm medium fast': 'Right-arm medium fast',
    'Right-arm Offbreak': 'Right-arm off-break',
    'Right-arm fast seam': 'Right-arm fast',
    'Right arm fast': 'Right-arm fast',
    'Right-arm fast': 'Right-arm fast',
    'Right-arm fast-medium/Off-spin': 'Right-arm fast-medium',
    'Right-arm off-break, Legbreak': 'Right-arm off-break and Legbreak',
    'Right-Arm Off Spin': 'Right-arm off-break',
    'Legbreak Googly': 'Right-arm leg-spin',  # Updated mapping
    'Righ-arm leg-spin': 'Right-arm leg-spin',
    'Left arm Medium': 'Left-arm medium',
    'Left-arm orthodox': 'Slow left-arm orthodox',
    'Left arm wrist spin': 'Left-arm wrist spin',
    'Right-arm off break': 'Right-arm off-break',
    'Righ-arm medium': 'Right-arm medium fast',  # Mapping to Right-arm medium fast
    'Right arm medium fast': 'Right-arm medium fast',  # Mapping to Right-arm medium fast
    'Right arm Medium': 'Right-arm medium fast',  # Mapping to Right-arm medium fast
}

# Apply the mapping to the 'bowling_style' column in the PDF dataframe
pdf['bowling_style'] = pdf['bowling_style'].replace(bowling_style_mapping)
# Sidebar for selecting between "Player Profile" and "Matchup Analysis"
sidebar_option = st.sidebar.radio(
    "Select an option:",
    ("Player Profile", "Matchup Analysis","Strength vs Weakness","Team Builder")
)

if sidebar_option == "Player Profile":
    st.header("Player Profile")

    # Player search input (selectbox)
    player_name = st.selectbox("Search for a player", idf['batsman'].unique())

    # Filter the data for the selected player
    player_info = idf[idf['batsman'] == player_name].iloc[0]

    # Check if the player exists in info_df
    matching_rows = info_df[info_df['Player_name'] == player_name]

    if not matching_rows.empty:
        # If there is a matching row, access the first one
        p_info = matching_rows.iloc[0]
    else:
        # st.write(f"No player found with the name '{player_name}'")
        p_info = None  # Set a fallback

    # Tabs for "Overview", "Career Statistics", and "Current Form"
    tab1, tab2, tab3 = st.tabs(["Overview", "Career Statistics", "Current Form"])

    with tab1:
        st.header("Overview")

        # Create columns for the first row (full name, country, age)
        col1, col2, col3 = st.columns(3)

        # Display player profile information
        with col1:
            st.markdown("FULL NAME:")
            st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{player_info['batsman']}</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("TEAM:")
            st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{p_info['team'].upper()}</span>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("AGE:")
            if p_info is not None:
                st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{p_info['age']}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)

        # Create columns for the second row (batting style, bowling style, playing role)
        col4, col5, col6 = st.columns(3)

        with col4:
            st.markdown("BATTING STYLE:")
            if p_info is not None:
                st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{p_info['batting_style'].upper()}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)

        # with col5:
        #     st.markdown("BOWLING STYLE:")
        #     if p_info is not None:
        #         if p_info['Bowling Style'] == 'N/A':
        #             st.markdown("<span style='font-size: 20px; font-weight: bold;'>NONE</span>", unsafe_allow_html=True)
        #         else:
        #             st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{p_info['bowling_style'].upper()}</span>", unsafe_allow_html=True)
        #     else:
        #         st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)
        with col5:
            st.markdown("BOWLING STYLE:")
            if p_info is not None:
                # Using .get() to safely access the 'Bowling Style' key
                bowling_style = p_info.get('Bowling Style', 'N/A')  # Default to 'N/A' if key doesn't exist
        
                if bowling_style == 'N/A':
                    st.markdown("<span style='font-size: 20px; font-weight: bold;'>NONE</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{bowling_style.upper()}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)


        with col6:
            st.markdown("PLAYING ROLE:")
            if p_info is not None:
                st.markdown(f"<span style='font-size: 20px; font-weight: bold;'>{p_info['role'].upper()}</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='font-size: 20px; font-weight: bold;'>N/A</span>", unsafe_allow_html=True)

    with tab2:
            st.header("Career Statistics")
    
            # Dropdown for Batting or Bowling selection
            option = st.selectbox("Select Career Stat Type", ("Batting", "Bowling"))
    
            # Show Career Averages based on the dropdown
            st.subheader("Career Performance")
    
            # Display Career Averages based on selection
            if option == "Batting":
                # Create a temporary DataFrame and filter the player's row
                temp_df = idf.drop(columns=['final_year','matches'])
                player_stats = temp_df[temp_df['batsman'] == player_name]  # Filter for the selected player
    
                # Convert column names to uppercase and replace underscores with spaces
                player_stats.columns = [col.upper().replace('_', ' ') for col in player_stats.columns]
                player_stats=round_up_floats(player_stats)
                # Display the player's statistics in a table format with bold headers
                st.markdown("### Batting Statistics")
                columns_to_convert = ['RUNS','HUNDREDS', 'FIFTIES','THIRTIES', 'HIGHEST SCORE']
    
                   # Fill NaN values with 0
                player_stats[columns_to_convert] = player_stats[columns_to_convert].fillna(0)
                    
                   # Convert the specified columns to integer type
                player_stats[columns_to_convert] = player_stats[columns_to_convert].astype(int)
                # player_stats=player_stats.drop(columns={'UNNAMED:0'})
                st.table(player_stats.style.set_table_attributes("style='font-weight: bold;'"))                
                # Initializing an empty DataFrame for results and a counter
                result_df = pd.DataFrame()
                i = 0
                
                # Checking if 'total_runs', 'batsman_runs', 'dismissal_kind', 'batsman', and 'over' are already in bpdf
                if 'total_runs' not in pdf.columns:
                    pdf['total_runs'] = pdf['runs_off_bat'] + pdf['extras']  # Create total_runs column
                
                    # Renaming necessary columns if they don't exist in the desired format
                    pdf = pdf.rename(columns={
                        'runs_off_bat': 'batsman_runs', 
                        'wicket_type': 'dismissal_kind', 
                        'striker': 'batsman', 
                        'innings': 'inning', 
                        'bowler': 'bowler_name'
                    })
                
                    # Drop rows where 'ball' is missing, if not already done
                    pdf = pdf.dropna(subset=['ball'])
                
                # Convert the 'ball' column to numeric if it's not already
                if not pd.api.types.is_numeric_dtype(pdf['ball']):
                    pdf['ball'] = pd.to_numeric(pdf['ball'], errors='coerce')
                
                # Calculate 'over' by applying lambda function (check if the 'over' column is already present)
                if 'over' not in pdf.columns:
                    pdf['over'] = pdf['ball'].apply(lambda x: mt.floor(x) + 1 if pd.notnull(x) else None)
                
                # Allowed states for batting analysis
                allowed_states = ['Andhra', 'Arunachal Pradesh', 'Assam', 'Baroda', 'Bengal',
                                  'Bihar', 'Chandigarh', 'Chattisgarh', 'Delhi', 'Goa', 'Gujarat',
                                  'Haryana', 'Himachal Pradesh', 'Hyderabad (India)',
                                  'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala',
                                  'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
                                  'Mumbai', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Railways',
                                  'Rajasthan', 'Saurashtra', 'Services', 'Sikkim', 'Tamil Nadu',
                                  'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'Vidarbha']
                
                # Creating a DataFrame to display venues and their corresponding states
                pdf['state'] = pdf['venue'].map(venue_state_map).fillna('Unknown')
                i = 0
                
                # Iterate over allowed states for batting analysis
                for state in allowed_states:
                    temp_df = pdf[pdf['batsman'] == player_name]  # Filter data for the selected batsman
                
                    # Filter for the specific state
                    temp_df = temp_df[temp_df['bowling_team'] == state]
                
                    # Apply the cumulative function (bcum)
                    temp_df = cumulator(temp_df)
                
                    # If the DataFrame is empty after applying `bcum`, skip this iteration
                    if temp_df.empty:
                        continue
                
                    # Add the state column with the current state's value
                    temp_df['opponent'] = state.upper()
                
                    # Reorder columns to make 'state' the first column
                    cols = temp_df.columns.tolist()
                    new_order = ['opponent'] + [col for col in cols if col != 'opponent']
                    temp_df = temp_df[new_order]
                
                    # Concatenate results into result_df
                    if i == 0:
                        result_df = temp_df
                        i += 1
                    else:
                        result_df = pd.concat([result_df, temp_df], ignore_index=True)
                
                # Display the final result_df
                result_df = result_df.drop(columns=['batsman', 'debut_year', 'final_year', 'batting_team'])
                result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                columns_to_convert = ['HUNDREDS', 'FIFTIES', 'THIRTIES', 'RUNS', 'HIGHEST SCORE']
                
                # Fill NaN values with 0
                result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
                
                # Convert the specified columns to integer type
                result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
                result_df = round_up_floats(result_df)
                cols = result_df.columns.tolist()
                
                # Specify the desired order with 'year' first
                new_order = ['OPPONENT', 'MATCHES'] + [col for col in cols if col not in ['MATCHES', 'OPPONENT']]
                
                # Reindex the DataFrame with the new column order
                result_df = result_df[new_order]
                
                st.markdown("### Opponentwise Performance")
                st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
                
                tdf = pdf[pdf['batsman'] == player_name]
                
                def standardize_season(season):
                    if '/' in season:  # Check if the season is in 'YYYY/YY' format
                        year = season.split('/')[0]  # Get the first part
                    else:
                        year = season  # Use as is if already in 'YYYY' format
                    return year.strip()  # Return the year stripped of whitespace
                
                tdf['season'] = tdf['season'].apply(standardize_season)
                
                # Populate an array of unique seasons
                unique_seasons = tdf['season'].unique()
                
                # Optional: Convert to a sorted list (if needed)
                unique_seasons = sorted(set(unique_seasons))
                tdf = pd.DataFrame(tdf)
                tdf['batsman_runs'] = tdf['batsman_runs'].astype(int)
                tdf['total_runs'] = tdf['total_runs'].astype(int)
                
                # Run a for loop and pass temp_df to a cumulative function
                i = 0
                for season in unique_seasons:
                    print(i)
                    temp_df = tdf[(tdf['season'] == season)]
                    print(temp_df.head())
                    temp_df = cumulator(temp_df)
                    if i == 0:
                        result_df = temp_df  # Initialize with the first result_df
                        i = 1 + i
                    else:
                        result_df = pd.concat([result_df, temp_df], ignore_index=True)
                    result_df = result_df.drop(columns=['batsman', 'debut_year','batting_team'])                  
                    # Convert specific columns to integers
                    # Round off the remaining float columns to 2 decimal places
                    float_cols = result_df.select_dtypes(include=['float']).columns
                    result_df[float_cols] = result_df[float_cols].round(2)
                
                result_df = result_df.rename(columns={'final_year': 'year'})
                result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                result_df = round_up_floats(result_df)
                columns_to_convert = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']
                
                # Fill NaN values with 0
                result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
                
                # Convert the specified columns to integer type
                result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
                
                # Display the results
                st.markdown(f"### **Yearwise Performance**")
                cols = result_df.columns.tolist()
                
                # Specify the desired order with 'year' first
                new_order = ['YEAR'] + [col for col in cols if col != 'YEAR']
                
                # Reindex the DataFrame with the new column order
                result_df = result_df[new_order]
                st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
                
                tdf = pdf[pdf['batsman'] == player_name]
                temp_df = tdf[(tdf['inning'] == 1)]
                temp_df = cumulator(temp_df)
                temp_df['inning'] = 1
                cols = temp_df.columns.tolist()
                new_order = ['inning'] + [col for col in cols if col != 'inning']
                # Reindex the DataFrame with the new column order
                temp_df = temp_df[new_order] 
                result_df = temp_df
                temp_df = tdf[(tdf['inning'] == 2)]
                temp_df = cumulator(temp_df)
                temp_df['inning'] = 2
                cols = temp_df.columns.tolist()
                new_order = ['inning'] + [col for col in cols if col != 'inning']
                # Reindex the DataFrame with the new column order
                temp_df = temp_df[new_order] 
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
                result_df = result_df.drop(columns=['batsman', 'debut_year', 'final_year', 'batting_team'])
                # Convert specific columns to integers
                # Round off the remaining float columns to 2 decimal places
                float_cols = result_df.select_dtypes(include=['float']).columns
                result_df[float_cols] = result_df[float_cols].round(2)
                
                result_df = result_df.rename(columns={'final_year': 'year'})
                result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                columns_to_convert = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']
                
                # Fill NaN values with 0
                result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
                
                # Convert the specified columns to integer type
                result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
                
                # Display the results
                result_df = result_df.drop(columns=['MATCHES'])
                st.markdown(f"### **Inningwise Performance**")
                st.table(result_df.reset_index(drop=True).style.set_table_attributes("style='font-weight: bold;'"))
                
                
                # # Creating a DataFrame to display venues and their corresponding countries
                pdf['state'] = pdf['venue'].map(venue_state_map).fillna('Unknown')
                allowed_state = ['Andhra', 'Arunachal Pradesh', 'Assam', 'Baroda', 'Bengal',
                                  'Bihar', 'Chandigarh', 'Chattisgarh', 'Delhi', 'Goa', 'Gujarat',
                                  'Haryana', 'Himachal Pradesh', 'Hyderabad (India)',
                                  'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala',
                                  'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
                                  'Mumbai', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Railways',
                                  'Rajasthan', 'Saurashtra', 'Services', 'Sikkim', 'Tamil Nadu',
                                  'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'Vidarbha']
                i=0
                
                for state in allowed_state:
                    temp_df = pdf[pdf['batsman'] == player_name]
                    # print(temp_df.match_id.unique())
                    # print(temp_df.head(20))
                    temp_df = temp_df[(temp_df['state'] == state)]
                    temp_df = cumulator(temp_df)
                    temp_df['state']=state.upper()
                    cols = temp_df.columns.tolist()
                    new_order = ['state'] + [col for col in cols if col != 'state']
                    # Reindex the DataFrame with the new column order
                    temp_df =temp_df[new_order]
                    # print(temp_df)
                 # If temp_df is empty after applying cumulator, skip to the next iteration
                    if len(temp_df) == 0:
                       continue
                    elif i==0:
                        result_df = temp_df
                        i=i+1
                    else:
                        result_df = result_df.reset_index(drop=True)
                        temp_df = temp_df.reset_index(drop=True)
                        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
                
                        result_df = pd.concat([result_df, temp_df],ignore_index=True)
                        
                
                    result_df = result_df.drop(columns=['batsman', 'debut_year', 'final_year', 'batting_team'])
                    # Round off the remaining float columns to 2 decimal places
                    float_cols = result_df.select_dtypes(include=['float']).columns
                    result_df[float_cols] = result_df[float_cols].round(2)
                result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                result_df = round_up_floats(result_df)
                columns_to_convert = ['RUNS', 'HUNDREDS', 'FIFTIES', 'THIRTIES', 'HIGHEST SCORE']
    
                #    # Fill NaN values with 0
                result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
                    
                #    # Convert the specified columns to integer type
                result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
                cols = result_df.columns.tolist()
                if 'STATE' in cols:
                    new_order = ['STATE'] + [col for col in cols if col != 'STATE']
                    result_df = result_df[new_order]
                # result_df = result_df.loc[:, ~result_df.columns.duplicated()]
                    result_df = result_df.drop(columns=['MATCHES'])
                st.markdown(f"### **In Host State**")
                st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))

            elif option == "Bowling":
                # Prepare the DataFrame for displaying player-specific bowling statistics
                temp_df = bidf
                    
                    # Filter for the selected player
                player_stats = temp_df[temp_df['bowler'] == player_name]  # Assuming bidf has bowler data
                if player_stats.empty:
                    st.markdown("No Bowling stats available")
                else:   
                        # Convert column names to uppercase and replace underscores with spaces
                        player_stats.columns = [col.upper().replace('_', ' ') for col in player_stats.columns]
                            
                            # Function to round float values if necessary (assuming round_up_floats exists)
                        player_stats = round_up_floats(player_stats)
                        columns_to_convert = ['RUNS','THREE WICKET HAULS', 'MAIDEN OVERS']
            
                        #    # Fill NaN values with 0
                        player_stats[columns_to_convert] =  player_stats[columns_to_convert].fillna(0)
                            
                        #    # Convert the specified columns to integer type
                        player_stats[columns_to_convert] =  player_stats[columns_to_convert].astype(int)
                            
                            # Display the player's bowling statistics in a table format with bold headers
                        player_stats = player_stats.drop(columns=['BOWLER'])
                        st.markdown("### Bowling Statistics")
                        st.table(player_stats.style.set_table_attributes("style='font-weight: bold;'"))  # Display the filtered DataFrame as a table
                        allowed_states = ['Andhra', 'Arunachal Pradesh', 'Assam', 'Baroda', 'Bengal',
                                          'Bihar', 'Chandigarh', 'Chattisgarh', 'Delhi', 'Goa', 'Gujarat',
                                          'Haryana', 'Himachal Pradesh', 'Hyderabad (India)',
                                          'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala',
                                          'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
                                          'Mumbai', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Railways',
                                          'Rajasthan', 'Saurashtra', 'Services', 'Sikkim', 'Tamil Nadu',
                                          'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'Vidarbha']
                        # Initializing an empty DataFrame for results and a counter
                        result_df = pd.DataFrame()
                        i = 0
                        
                        # Checking if 'total_runs', 'batsman_runs', 'dismissal_kind', 'batsman', and 'over' are already in bpdf
                        if 'total_runs' not in bpdf.columns:
                            bpdf['total_runs'] = bpdf['runs_off_bat'] + bpdf['extras']  # Create total_runs column
                        
                            # Renaming necessary columns if they don't exist in the desired format
                            bpdf = bpdf.rename(columns={
                                'runs_off_bat': 'batsman_runs', 
                                'wicket_type': 'dismissal_kind', 
                                'striker': 'batsman', 
                                'innings': 'inning', 
                                'bowler': 'bowler_name'
                            })
                            # Drop rows where 'ball' is missing, if not already done
                            bpdf = bpdf.dropna(subset=['ball'])
                        
                        # Convert the 'ball' column to numeric if it's not already
                        if not pd.api.types.is_numeric_dtype(bpdf['ball']):
                            bpdf['ball'] = pd.to_numeric(bpdf['ball'], errors='coerce')
                        
                        # Calculate 'over' by applying lambda function (check if the 'over' column is already present)
                        if 'over' not in bpdf.columns:
                            bpdf['over'] = bpdf['ball'].apply(lambda x: mt.floor(x) + 1 if pd.notnull(x) else None)
                        st.markdown("### Opponentwise Performance")
                        for country in allowed_states:
                                    # Iterate over allowed countries for batting analysis
                                    temp_df = bpdf[bpdf['bowler'] == player_name]  # Filter data for the selected batsman
                                        
                                    # Filter for the specific country
                                    temp_df = temp_df[temp_df['batting_team'] == country]
                            
                                    # Apply the cumulative function (bcum)
                                    temp_df = bcum(temp_df)
                                
                                    # If the DataFrame is empty after applying `bcum`, skip this iteration
                                    if temp_df.empty:
                                        continue
                                
                                    # Add the country column with the current country's value
                                    temp_df['opponent'] = country.upper()
                                
                                    # Reorder columns to make 'country' the first column
                                    cols = temp_df.columns.tolist()
                                    new_order = ['opponent'] + [col for col in cols if col != 'opponent']
                                    temp_df = temp_df[new_order]
                                    
                                
                                    # Concatenate results into result_df
                                    if i == 0:
                                        result_df = temp_df
                                        i += 1
                                    else:
                                        result_df = pd.concat([result_df, temp_df], ignore_index=True)
                        # Display the final result_df
                        result_df = result_df.drop(columns=['bowler','debut_year','final_year'])
                        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                        columns_to_convert = ['RUNS','THREE WICKET HAULS', 'MAIDEN OVERS']
            
                           # Fill NaN values with 0
                        result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
                            
                           # Convert the specified columns to integer type
                        result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
                        # result_df=round_up_floats(result_df)
                        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
              
                        
                        tdf = bpdf[bpdf['bowler'] == player_name]  # Filter data for the specific bowler
            
                        def standardize_season(season):
                                        if '/' in season:  # Check if the season is in 'YYYY/YY' format
                                            year = season.split('/')[0]  # Get the first part
                                        else:
                                            year = season  # Use as is if already in 'YYYY' format
                                        return year.strip()  # Return the year stripped of whitespace
                                    # Standardize the 'season' column
                        tdf['season'] = tdf['season'].apply(standardize_season)
                        
                                    # Populate an array of unique seasons
                        unique_seasons = sorted(set(tdf['season'].unique()))  # Optional: Sorted list of unique seasons
                        
                                    # Initialize an empty DataFrame to store the final results
                        i = 0
                        for season in unique_seasons:
                                temp_df = tdf[tdf['season'] == season]  # Filter data for the current season
                                temp_df = bcum(temp_df)  # Apply the cumulative function (specific to your logic)
                                temp_df['YEAR'] = season
                                    
                                if i == 0:
                                        result_df = temp_df  # Initialize the result_df with the first season's data
                                        i += 1
                                else:
                                        result_df = pd.concat([result_df, temp_df], ignore_index=True)  # Append subsequent data
                                          
                        result_df = result_df.drop(columns=['bowler','debut_year','final_year'])
                        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                        columns_to_convert = ['THREE WICKET HAULS', 'MAIDEN OVERS']
    
                           # Fill NaN values with 0
                        result_df[columns_to_convert] = result_df[columns_to_convert].fillna(0)
                
                           # Convert the specified columns to integer type
                        result_df[columns_to_convert] = result_df[columns_to_convert].astype(int)
                        result_df=round_up_floats(result_df)
                        # result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
            
                        # No need to convert columns to integer (for bowling-specific data)
            
                        # Display the results
                        st.markdown(f"### **Yearwise Bowling Performance**")
                        cols = result_df.columns.tolist()
            
                        # Specify the desired order with 'YEAR' first
                        new_order = ['YEAR'] + [col for col in cols if col != 'YEAR']
            
                        # Reindex the DataFrame with the new column order
                        result_df = result_df[new_order]
            
                        # Display the table with bold headers
                        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
                        # Filter data for the specific bowler
                        tdf = bpdf[bpdf['bowler'] == player_name]   
                        
                        # Process for the first inning
                        temp_df = tdf[(tdf['inning'] == 1)]
                        temp_df = bcum(temp_df)  # Apply the cumulative function specific to bowlers
                        temp_df['inning'] = 1  # Add the inning number
            
                        # Reorder columns to have 'inning' first
                        cols = temp_df.columns.tolist()
                        new_order = ['inning'] + [col for col in cols if col != 'inning']          
                        temp_df = temp_df[new_order] 
            
                        # Initialize result_df with the first inning's data
                        result_df = temp_df
            
                        # Process for the second inning
                        temp_df = tdf[(tdf['inning'] == 2)]
                        temp_df = bcum(temp_df)  # Apply the cumulative function specific to bowlers
                        temp_df['inning'] = 2  # Add the inning number
            
                        # Reorder columns to have 'inning' first
                        cols = temp_df.columns.tolist()
                        new_order = ['inning'] + [col for col in cols if col != 'inning']          
                        temp_df = temp_df[new_order] 
            
                        # Concatenate the results for both innings
                        result_df = pd.concat([result_df, temp_df], ignore_index=True)
            
                        # Drop unnecessary columns
                        result_df = result_df.drop(columns=['bowler','debut_year','final_year'])
                        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                        columns_to_convert = ['THREE WICKET HAULS', 'MAIDEN OVERS']
    
                           # Fill NaN values with 0
                        result_df[columns_to_convert] =  result_df[columns_to_convert].fillna(0)
                
                           # Convert the specified columns to integer type
                        result_df[columns_to_convert] =  result_df[columns_to_convert].astype(int)
                        result_df=round_up_floats(result_df)
            
                        # Display the results
                        st.markdown(f"### **Inningwise Bowling Performance**")
                        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))
    
            
            
                        # Creating a DataFrame to display venues and their corresponding countries
                        bpdf['state'] = bpdf['venue'].map(venue_state_map)
                        allowed_states = ['Andhra', 'Arunachal Pradesh', 'Assam', 'Baroda', 'Bengal',
                                          'Bihar', 'Chandigarh', 'Chattisgarh', 'Delhi', 'Goa', 'Gujarat',
                                          'Haryana', 'Himachal Pradesh', 'Hyderabad (India)',
                                          'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala',
                                          'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
                                          'Mumbai', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Railways',
                                          'Rajasthan', 'Saurashtra', 'Services', 'Sikkim', 'Tamil Nadu',
                                          'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'Vidarbha']
                        i = 0
                        for country in allowed_states:
                            temp_df = bpdf[bpdf['bowler'] == player_name] 
                            temp_df = temp_df[(temp_df['country'] == country)]
                            temp_df = bcum(temp_df)
                            temp_df.insert(0, 'country', country.upper())
                
            
                            # If temp_df is empty after applying bcum, skip to the next iteration
                            if len(temp_df) == 0:
                                continue
                            elif i == 0:
                                result_df = temp_df
                                i += 1
                            else:
                                result_df = result_df.reset_index(drop=True)
                                temp_df = temp_df.reset_index(drop=True)
                                result_df = result_df.loc[:, ~result_df.columns.duplicated()]
            
                                result_df = pd.concat([result_df, temp_df], ignore_index=True)
            
                        if 'bowler' in result_df.columns:
                            result_df = result_df.drop(columns=['bowler','debut_year','final_year'])
                            result_df=result_df.rename(columns={'country':'state'})
                        result_df.columns = [col.upper().replace('_', ' ') for col in result_df.columns]
                        columns_to_convert = ['RUNS','THREE WICKET HAULS', 'MAIDEN OVERS']
    
                           # Fill NaN values with 0
                        result_df[columns_to_convert] =  result_df[columns_to_convert].fillna(0)
                
                           # Convert the specified columns to integer type
                        result_df[columns_to_convert] =  result_df[columns_to_convert].astype(int)
                        result_df=round_up_floats(result_df)
            
                        st.markdown(f"### **In Host State**")
                        st.table(result_df.style.set_table_attributes("style='font-weight: bold;'"))


                  
    
