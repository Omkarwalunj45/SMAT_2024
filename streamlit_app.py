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
bpdf=pdf
idf = pd.read_csv("Dataset/lifesaver_bat_smat.csv",low_memory=False)
info_df=pd.read_csv("Dataset/cricket_players_data.csv",low_memory=False)
bidf=pd.read_csv("Dataset/lifesaver_bowl_smat.csv",low_memory=False)
info_df=info_df.rename(columns={'player':'Player_name'})
pdf[['noballs', 'wides','byes','legbyes','penalty']] = pdf[['noballs', 'wides','byes','legbyes','penalty']].fillna(0).astype(int)
pdf['valid_ball'] = pdf.apply(lambda x: 1 if (x['wides'] == 0 and x['noballs'] == 0) else 0, axis=1)

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
