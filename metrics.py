from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def intra_list_diversity(recommendations: pd.DataFrame, feature_cols: list) -> float:
    """
    Estimate the average pairwise Euclidean distance between items in user recommendations.
    
    Parameters:
    recommendations : pd.DataFrame
        DataFrame containing song recommendations.
    feature_cols : list
        List of feature column names used to calculate Euclidean distance.
    
    Returns:
    float
        Average pairwise Euclidean distance (Intra-List Diversity score).
    """
    distances = []
    for i in range(len(recommendations)):
        for j in range(i + 1, len(recommendations)):
            song1 = recommendations.iloc[i][feature_cols]
            song2 = recommendations.iloc[j][feature_cols]
            distances.append(euclidean(song1, song2))
    
    return np.mean(distances) if distances else 0

def genre_coverage(recommendations: pd.DataFrame, df: pd.DataFrame) -> float:
    """
    Calculate the proportion of unique genres in recommendations compared to the dataset.
    
    Parameters:
    recommendations : pd.DataFrame
        DataFrame containing song recommendations.
    df : pd.DataFrame
        Full dataset to determine total genre diversity.
    
    Returns:
    float
        Genre coverage percentage (0-1 range).
    """
    unique_genres = recommendations['track_genre'].nunique()
    total_genres = df['track_genre'].nunique()
    return unique_genres / total_genres if total_genres > 0 else 0

def feature_variance(recommendations: pd.DataFrame, feature_cols: list) -> float:
    """
    Compute the mean variance of selected features in the recommendations.
    
    Parameters:
    recommendations : pd.DataFrame
        DataFrame containing song recommendations.
    feature_cols : list
        List of feature columns to measure variance.
    
    Returns:
    float
        Mean variance of the selected features.
    """
    return recommendations[feature_cols].var().mean()

def compare_recommendation_systems(clustering_recs: pd.DataFrame, graph_recs: pd.DataFrame):
    """
    Compare the similarity between recommendations from two systems: 
    1. Clustering model
    2. Graph model
    
    
    Parameters:
    -------------
    clustering_recs: pd.DataFrame
        DataFrame containing 15 song recommendations from generate_playlist 
    graph_recs: pd.DataFrame
        DataFrame containing 15 song recommendations from the music knowledge graph (NEED OUTPUT FROM GRAPH STILL)
    
    Returns:
    -------------
    dict
        Dictionary containing similarity scores between the recommendation systems
    """
    # Ensure both recommendation sets have min 15 songs
    clustering_recs = clustering_recs.iloc[:15]
    graph_recs = graph_recs.iloc[:15]
    
    feature_cols = ['danceability', 'energy', 'valence', 'tempo']
    
    # Convert to numpy arrays
    clustering_features = clustering_recs[feature_cols].to_numpy()
    graph_features = graph_recs[feature_cols].to_numpy()
    
    # Compute similarity between the two recommendation lists
    similarity_score = cosine_similarity(clustering_features, graph_features).mean()
    
    return {
        'clustering_vs_graph': similarity_score,
    }


