'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd
import ast  
def load_data():
    '''
    Load data from CSV files

    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    model_pred_df = pd.read_csv('data/prediction_model_03.csv')
    genres_df = pd.read_csv('data/genres.csv')
    return model_pred_df, genres_df

def process_data(model_pred_df, genres_df):
    import ast
'''
    Process data to get genre lists and count dictionaries

    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''
genre_list = genres_df['genre'].tolist()
genre_true_counts = genre_tp_counts = genre_fp_counts = {g: 0 for g in genre_list}

for _, row in model_pred_df.iterrows():
predicted = row['predicted']
actual = ast.literal_eval(row['actual genres'])

for g in actual:
if g in genre_true_counts:
     genre_true_counts[g] += 1

if predicted in genre_tp_counts if row['correct?'] else genre_fp_counts:
(genre_tp_counts if row['correct?'] else genre_fp_counts)[predicted] += 1
return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
