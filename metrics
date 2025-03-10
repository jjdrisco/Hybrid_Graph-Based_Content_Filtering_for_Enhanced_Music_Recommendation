from scipy.spatial.distance import euclidean
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