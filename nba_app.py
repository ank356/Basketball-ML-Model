import streamlit as st # For AI interactive data apps
import pandas as pd
import joblib
import numpy as np
import google.generativeai as genai
from nba_api.stats.static import teams

st.set_page_config(page_title="NBA AI Predictor", page_icon="🏀")
st.title("AI NBA Predictor Chatbot")

API_KEY = "AIzaSyAgig_w88_m29qSRKguRHmXtjtEWCV-mEk" 
genai.configure(api_key=API_KEY)
llm_model = genai.GenerativeModel('gemini-2.5-flash')

@st.cache_resource
def load_assets():
    model = joblib.load('nba_model.pkl')
    encoder = joblib.load('label_encoder.pkl')
    df = pd.read_csv('nba_historical_data.csv')
    return model, encoder, df

rf_model, le, history_df = load_assets()

nba_teams = teams.get_teams()
name_to_id = {team['full_name']: team['id'] for team in nba_teams}
team_names = sorted(list(name_to_id.keys()))

def get_matchup_features(home_team_name, away_team_name):

    raw_home_id = name_to_id[home_team_name]
    raw_away_id = name_to_id[away_team_name]

    home_id = le.transform([raw_home_id])[0]
    away_id= le.transform([raw_away_id])[0]

    home_games = history_df[history_df["TEAM_ID"] == home_id].tail(5)
    
    features = np.array([[
        home_id, # TEAM_ID
        away_id, # OPP_TEAM_ID
        1, # HGA (1 because they are the home team)
        home_games["WIN"].iloc[-1], # LAST_GAME_OUTCOME
        home_games["PTS"].mean(), # PTS_5G_AVG
        home_games["OREB"].mean(), # OREB_5G_AVG
        home_games["DREB"].mean(), # DREB_5G_AVG
        home_games["REB"].mean(), # REB_5G_AVG
        home_games["AST"].mean(), # AST_5G_AVG
        home_games["STL"].mean(), # STL_5G_AVG
        home_games["BLK"].mean(), # BLK_5G_AVG
        home_games["TOV"].mean(), # TOV_5G_AVG
        home_games["EFG%"].mean(), # EFG%_5G_AVG
        home_games["TOV%"].mean(), # TOV%_5G_AVG
        home_games["FTR"].mean(), # FTR_5G_AVG
        home_games["TS%"].mean() # TS%_5G_AVG
    ]])
    
    return features 

st.write("See who will win tonight's game...")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", team_names)
with col2:
    away_team = st.selectbox("Away Team", team_names)

if st.button("Predict Outcome"):
    
    if home_team == away_team:
        st.error("Please input two different teams.")
    else:
        with st.spinner("Analyzing stats..."):
            
            matchup_data = get_matchup_features(home_team, away_team)
            
            prediction_probs = rf_model.predict_proba(matchup_data)[0]
            win_prob = round(prediction_probs[1] * 100, 2)
            
            if win_prob > 50:
                predicted_winner = home_team 
                confidence = win_prob
            else:
                predicted_winner = away_team
                confidence = round(100 - win_prob, 2)

            prompt = f"""
            A user wants to know who will win tonight's game between the {home_team} (Home) and the {away_team} (Away). 
            
            My highly accurate machine learning model predicts the {predicted_winner} will win 
            with a {confidence}% confidence based on their last 15 games played. 
            
            Write a short 1-paragraph response explaining this prediction to the user. 
            Talk seriously and emphasize some main statistics and numbers that helped you determine the outcome.
            """
            response = llm_model.generate_content(prompt)
            
            st.success(f"**Model Prediction:** {predicted_winner} with {confidence}% confidence")
            st.write("---")
            st.write(response.text)