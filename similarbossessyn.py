import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import requests
import streamlit as st

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('clean_NBA.csv')
    return df

# Define columns for each model
features_model1 = [
    'TS%', 'eFG%', 'FTRate', 'OR%', 'DR%', 'AST%', 'TO%', 'STL%', 'BLK%', 'Usg%', '+/-', 'PPP',
    'PTS', '2P%', '%2P', 'FGM', 'FGA', 'FGM-A', 'FG%', '3PM', '3PA', '3PM-A', '3P-R', '2PM', '2PA', '2PM-A',
    'FTM', 'FTA', 'FTM-A', 'ORB', 'DRB', 'REB', 'AST', 'TO', '3P%', '%3P', 'STL', 'FT%'
]

numeric_columns_model2 = [
    'G', 'GS', 'MIN', 'TS%', 'eFG%', 'FTRate', 'OR%', 'DR%', 'AST%', 'TO%', 'STL%', 'BLK%', 'Usg%', '+/-',
    'PPP', 'PTS', '2P%', '%2P', 'FGM', 'FGA', 'FGM-A', 'FG%', '3PM', '3PA', '3PM-A', '3P-R',
    '2PM', '2PA', '2PM-A', 'FTM', 'FTA', 'FTM-A', 'ORB', 'DRB', 'REB', 'AST', 'TO', '3P%', '%3P', 'STL', 'FT%',
    'Spot-Up_count', 'Spot-Up_percent', 'Spot-Up_ppp', 'Cut_count', 'Cut_percent', 'Cut_ppp', 'Transition_count',
    'Transition_percent', 'Transition_ppp', 'P&R Roll Man_count', 'P&R Roll Man_percent', 'P&R Roll Man_ppp',
    'Off Screen_count', 'Off Screen_percent', 'Off Screen_ppp', 'Hand Off_count', 'Hand Off_percent', 'Hand Off_ppp',
    'P&R Ball Handler_count', 'P&R Ball Handler_percent', 'P&R Ball Handler_ppp', 'Offensive Rebound_count',
    'Offensive Rebound_percent', 'Offensive Rebound_ppp', 'ISO_count', 'ISO_percent', 'ISO_ppp', 'No Play Type_count',
    'No Play Type_percent', 'No Play Type_ppp'
]

numeric_columns_model3 = [
    'Spot-Up_count', 'Spot-Up_percent', 'Spot-Up_ppp',
    'Cut_count', 'Cut_percent', 'Cut_ppp',
    'Transition_count', 'Transition_percent', 'Transition_ppp',
    'P&R Roll Man_count', 'P&R Roll Man_percent', 'P&R Roll Man_ppp',
    'Off Screen_count', 'Off Screen_percent', 'Off Screen_ppp',
    'Hand Off_count', 'Hand Off_percent', 'Hand Off_ppp',
    'P&R Ball Handler_count', 'P&R Ball Handler_percent', 'P&R Ball Handler_ppp',
    'Offensive Rebound_count', 'Offensive Rebound_percent', 'Offensive Rebound_ppp',
    'ISO_count', 'ISO_percent', 'ISO_ppp',
    'No Play Type_count', 'No Play Type_percent', 'No Play Type_ppp'
]

# Handle columns with hyphens (e.g., 'FGM-A', '3PM-A', '2PM-A', 'FTM-A')
def preprocess_data(df, columns):
    for col in ['FGM-A', '3PM-A', '2PM-A', 'FTM-A']:
        if col in df.columns:
            # Split the column into two new columns
            df[[f'{col}_made', f'{col}_attempted']] = df[col].str.split('-', expand=True).astype(float)
            # Drop the original column
            df.drop(columns=[col], inplace=True)
            # Add the new columns to the columns list
            if col in columns:
                columns.remove(col)
                columns.extend([f'{col}_made', f'{col}_attempted'])
    return df

# Normalize the features
def normalize_features(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

# Check for missing values and handle them
def handle_missing_values(df, columns):
    if df[columns].isnull().any().any():
        st.warning("The dataset contains missing values. Filling missing values with 0.")
        df[columns] = df[columns].fillna(0)
    return df

# Calculate cosine similarity with weights
def calculate_similarity(df, columns, weights=None):
    if weights:
        weighted_df = df[columns].copy()
        for col, weight in weights.items():
            if col in weighted_df.columns:
                weighted_df[col] *= weight
        similarity_matrix = cosine_similarity(weighted_df)
    else:
        similarity_matrix = cosine_similarity(df[columns])
    similarity_df = pd.DataFrame(similarity_matrix, index=df['player_id'], columns=df['player_id'])
    return similarity_df

# Fetch player names from the API
def fetch_player_names(player_ids, token):
    headers = {"Authorization": f"Bearer {token}"}
    id_string = ','.join(map(str, player_ids))
    api_url = f"https://stats.fastmodelsports.com/query/players/{id_string}?league=NBA&season=24"
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return {player['id']: player['name'] for player in data}
    else:
        st.error(f"Failed to fetch data: {response.status_code}")
        return {}

# Find top similar players
def find_top_similar_players(player_id, similarity_df, df, columns, player_names, top_n=5):
    if player_id not in similarity_df.columns:
        raise ValueError(f"Player with player_id '{player_id}' not found in the dataset.")
    
    player_similarity_scores = similarity_df[player_id]
    top_similar_players = player_similarity_scores.sort_values(ascending=False).iloc[1:top_n + 1]
    
    player_features = df[df['player_id'] == player_id][columns].values.flatten()
    
    results = []
    for similar_player_id, similarity_score in top_similar_players.items():
        similar_player_features = df[df['player_id'] == similar_player_id][columns].values.flatten()
        differences = abs(player_features - similar_player_features)
        similarity_percentages = (1 - differences) * 100
        similarity_metrics = pd.DataFrame({
            'Feature': columns,
            'Similarity_Percentage': similarity_percentages
        })
        similarity_metrics = similarity_metrics.sort_values(by='Similarity_Percentage', ascending=False).head(5)
        results.append({
            'player_id': similar_player_id,
            'player_name': player_names.get(similar_player_id, 'Unknown'),
            'similarity_score': similarity_score,
            'top_metrics': similarity_metrics
        })
    
    return results, player_names.get(player_id, 'Unknown')

# Streamlit App
def main():
    st.title("NBA Player Similarity Analysis")
    
    # Load data
    df = load_data()
    
    # Sidebar for model selection
    st.sidebar.title("Model Selection")
    model = st.sidebar.radio("Choose a model", ["Model 1: Basic Stats", "Model 2: Advanced Stats", "Model 3: Play Types", "Comparative Model"])
    
    # Bearer Token Input
    token = st.sidebar.text_input("Enter your Bearer Token", type="password")
    
    if token:
        # Fetch player names
        player_names = fetch_player_names(df['player_id'].tolist(), token)
        
        # Searchable Player Database
        st.sidebar.subheader("Search for a Player")
        search_query = st.sidebar.text_input("Type a player's name to search", "")
        
        if search_query:
            # Filter players based on search query
            filtered_players = [name for name in player_names.values() if search_query.lower() in name.lower()]
            selected_player_name = st.sidebar.selectbox("Select a player", filtered_players)
            
            if selected_player_name:
                # Get player ID from name
                selected_player_id = [id for id, name in player_names.items() if name == selected_player_name][0]
                
                if model == "Model 1: Basic Stats":
                    st.header("Model 1: BoxScores Stats")
                    df_model1 = preprocess_data(df.copy(), features_model1)
                    df_model1 = handle_missing_values(df_model1, features_model1)
                    df_model1 = normalize_features(df_model1, features_model1)
                    similarity_df_model1 = calculate_similarity(df_model1, features_model1)
                    try:
                        top_similar_players, player_name = find_top_similar_players(
                            selected_player_id, similarity_df_model1, df_model1, features_model1, player_names, top_n=5
                        )
                        display_results(top_similar_players, player_name)
                    except ValueError as e:
                        st.error(str(e))
                
                elif model == "Model 2: Advanced Stats":
                    st.header("Model 2: BoxScore + Synergy  Stats")
                    df_model2 = preprocess_data(df.copy(), numeric_columns_model2)
                    df_model2 = handle_missing_values(df_model2, numeric_columns_model2)
                    df_model2 = normalize_features(df_model2, numeric_columns_model2)
                    weights_model2 = {
                        'PTS': 2.0, 'AST': 1.5, 'REB': 1.5, '3P%': 1.2, 'FG%': 1.2, 'FT%': 1.0,
                        'STL': 1.0, 'BLK': 1.0, 'TO%': -1.0, 'Usg%': 1.0, '+/-': 1.5, 'PPP': 1.5,
                        'Spot-Up_ppp': 1.0, 'Cut_ppp': 1.0, 'Transition_ppp': 1.0, 'P&R Roll Man_ppp': 1.0,
                        'Off Screen_ppp': 1.0, 'Hand Off_ppp': 1.0, 'P&R Ball Handler_ppp': 1.0,
                        'Offensive Rebound_ppp': 1.0, 'ISO_ppp': 1.0, 'No Play Type_ppp': 1.0
                    }
                    similarity_df_model2 = calculate_similarity(df_model2, numeric_columns_model2, weights_model2)
                    try:
                        top_similar_players, player_name = find_top_similar_players(
                            selected_player_id, similarity_df_model2, df_model2, numeric_columns_model2, player_names, top_n=5
                        )
                        display_results(top_similar_players, player_name)
                    except ValueError as e:
                        st.error(str(e))
                
                elif model == "Model 3: Play Types":
                    st.header("Model 3: Tendencies ")
                    df_model3 = preprocess_data(df.copy(), numeric_columns_model3)
                    df_model3 = handle_missing_values(df_model3, numeric_columns_model3)
                    df_model3 = normalize_features(df_model3, numeric_columns_model3)
                    weights_model3 = {
                        'Spot-Up_ppp': 1.0, 'Cut_ppp': 1.0, 'Transition_ppp': 1.0,
                        'P&R Roll Man_ppp': 1.0, 'Off Screen_ppp': 1.0, 'Hand Off_ppp': 1.0,
                        'P&R Ball Handler_ppp': 1.0, 'Offensive Rebound_ppp': 1.0,
                        'ISO_ppp': 1.0, 'No Play Type_ppp': 1.0
                    }
                    similarity_df_model3 = calculate_similarity(df_model3, numeric_columns_model3, weights_model3)
                    try:
                        top_similar_players, player_name = find_top_similar_players(
                            selected_player_id, similarity_df_model3, df_model3, numeric_columns_model3, player_names, top_n=5
                        )
                        display_results(top_similar_players, player_name)
                    except ValueError as e:
                        st.error(str(e))
                
                elif model == "Comparative Model":
                    st.header("Comparative Model")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Model 1: Basic Stats")
                        df_model1 = preprocess_data(df.copy(), features_model1)
                        df_model1 = handle_missing_values(df_model1, features_model1)
                        df_model1 = normalize_features(df_model1, features_model1)
                        similarity_df_model1 = calculate_similarity(df_model1, features_model1)
                        try:
                            top_similar_players, player_name = find_top_similar_players(
                                selected_player_id, similarity_df_model1, df_model1, features_model1, player_names, top_n=5
                            )
                            display_results(top_similar_players, player_name)
                        except ValueError as e:
                            st.error(str(e))
                    
                    with col2:
                        st.subheader("Model 2: Advanced Stats")
                        df_model2 = preprocess_data(df.copy(), numeric_columns_model2)
                        df_model2 = handle_missing_values(df_model2, numeric_columns_model2)
                        df_model2 = normalize_features(df_model2, numeric_columns_model2)
                        weights_model2 = {
                            'PTS': 2.0, 'AST': 1.5, 'REB': 1.5, '3P%': 1.2, 'FG%': 1.2, 'FT%': 1.0,
                            'STL': 1.0, 'BLK': 1.0, 'TO%': -1.0, 'Usg%': 1.0, '+/-': 1.5, 'PPP': 1.5,
                            'Spot-Up_ppp': 1.0, 'Cut_ppp': 1.0, 'Transition_ppp': 1.0, 'P&R Roll Man_ppp': 1.0,
                            'Off Screen_ppp': 1.0, 'Hand Off_ppp': 1.0, 'P&R Ball Handler_ppp': 1.0,
                            'Offensive Rebound_ppp': 1.0, 'ISO_ppp': 1.0, 'No Play Type_ppp': 1.0
                        }
                        similarity_df_model2 = calculate_similarity(df_model2, numeric_columns_model2, weights_model2)
                        try:
                            top_similar_players, player_name = find_top_similar_players(
                                selected_player_id, similarity_df_model2, df_model2, numeric_columns_model2, player_names, top_n=5
                            )
                            display_results(top_similar_players, player_name)
                        except ValueError as e:
                            st.error(str(e))
                    
                    with col3:
                        st.subheader("Model 3: Play Types")
                        df_model3 = preprocess_data(df.copy(), numeric_columns_model3)
                        df_model3 = handle_missing_values(df_model3, numeric_columns_model3)
                        df_model3 = normalize_features(df_model3, numeric_columns_model3)
                        weights_model3 = {
                            'Spot-Up_ppp': 1.0, 'Cut_ppp': 1.0, 'Transition_ppp': 1.0,
                            'P&R Roll Man_ppp': 1.0, 'Off Screen_ppp': 1.0, 'Hand Off_ppp': 1.0,
                            'P&R Ball Handler_ppp': 1.0, 'Offensive Rebound_ppp': 1.0,
                            'ISO_ppp': 1.0, 'No Play Type_ppp': 1.0
                        }
                        similarity_df_model3 = calculate_similarity(df_model3, numeric_columns_model3, weights_model3)
                        try:
                            top_similar_players, player_name = find_top_similar_players(
                                selected_player_id, similarity_df_model3, df_model3, numeric_columns_model3, player_names, top_n=5
                            )
                            display_results(top_similar_players, player_name)
                        except ValueError as e:
                            st.error(str(e))

# Display results
def display_results(top_similar_players, player_name):
    st.subheader(f"Top 5 Similar Players to {player_name}")
    for result in top_similar_players:
        st.write(f"**Player Name:** {result['player_name']} (ID: {result['player_id']})")
        st.write(f"**Overall Similarity Score:** {result['similarity_score']:.4f}")
        st.write("**Top 5 Metrics Contributing to Similarity:**")
        st.dataframe(result['top_metrics'])

# Run the app
if __name__ == "__main__":
    main()