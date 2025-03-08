# %%
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import concurrent.futures
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, Tuple, List, Any
import time
from datetime import datetime

# %%
def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load and prepare the music dataset
    
    Parameters:
    filepath (str): Path to the CSV file containing music features
    
    Returns:
    pd.DataFrame: Processed DataFrame
    """
    df = pd.read_csv(filepath)
    return df

# %%
def split_data_by_genre(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split the dataset by genre
    
    Parameters:
    df (pd.DataFrame): DataFrame containing music data with genre column
    
    Returns:
    dict: Dictionary mapping genres to their respective DataFrames
    """
    # Assume 'genre' column exists, adjust if your column name is different
    genres = df['track_genre'].unique()
    genre_dfs = {}
    
    for genre in genres:
        genre_dfs[genre] = df[df['track_genre'] == genre]
    
    return genre_dfs

# %%
def cluster_music_data(data: pd.DataFrame, n_clusters: int = 50) -> Tuple[pd.DataFrame, np.ndarray, MiniBatchKMeans]:
    """
    Function to perform clustering on music feature data
    
    Parameters:
    data (pd.DataFrame): DataFrame containing music features
    n_clusters (int): Number of clusters to form
    
    Returns:
    tuple: (DataFrame with data, cluster labels, clustering model)
    """
    # Subset columns for clustering
    X = data[['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 
              'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
              'valence', 'tempo', 'time_signature', 'encoded_genre']]
    X = X.dropna()
    
    if len(X) < n_clusters:
        # Adjust n_clusters if there are fewer samples than requested clusters
        n_clusters = max(1, len(X) // 10)  # Ensure at least 1 cluster
    
    # Initialize and fit MiniBatch K-Means clustering
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=0,
        max_iter=10,
        n_init="auto",
        reassignment_ratio=0.1
    ).fit(X)
    
    # Predict clusters
    labels = kmeans.predict(X)
    
    return X, labels, kmeans

# %%
def save_compressed_model(model: Any, genre: str, output_dir: str = "models", compression_level: int = 6) -> str:
    """
    Save a compressed clustering model to disk using gzip
    
    Parameters:
    model: The clustering model to save
    genre (str): Genre label for the model
    output_dir (str): Directory to save the model
    compression_level (int): Compression level (1-9, where 9 is maximum compression)
    
    Returns:
    str: Path to the saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"cluster_model_{genre}.pkl.gz")
    
    with gzip.open(filepath, 'wb', compresslevel=compression_level) as f:
        pickle.dump(model, f)
        
    return filepath

# %%
def load_compressed_model(genre: str, model_dir: str = "models") -> Any:
    """
    Load a compressed clustering model from disk
    
    Parameters:
    genre (str): Genre label for the model to load
    model_dir (str): Directory containing saved models
    
    Returns:
    The loaded clustering model
    """
    filepath = os.path.join(model_dir, f"cluster_model_{genre}.pkl.gz")
    
    with gzip.open(filepath, 'rb') as f:
        model = pickle.load(f)
        
    return model

# %%
def process_genre(genre: str, genre_df: pd.DataFrame, output_dir: str) -> Dict:
    """
    Process a single genre for multithreaded execution
    
    Parameters:
    genre (str): Genre name
    genre_df (pd.DataFrame): DataFrame filtered for this genre
    output_dir (str): Directory to save models
    
    Returns:
    dict: Results for this genre
    """
    print(f"Processing genre: {genre} with {len(genre_df)} samples")
    
    # Adjust n_clusters based on dataset size
    n_clusters = min(50, max(5, len(genre_df) // 20))
    
    # Cluster the data
    X, labels, model = cluster_music_data(genre_df, n_clusters=n_clusters)
    
    # Save the model with compression
    model_path = save_compressed_model(model, genre, output_dir, compression_level=6)
    
    # Calculate model file size
    model_size = os.path.getsize(model_path) / 1024  # Size in KB
    
    # Return results
    return {
        "genre": genre,
        "data_shape": X.shape,
        "num_clusters": n_clusters,
        "model_path": model_path,
        "model_size_kb": model_size,
        "cluster_distribution": np.bincount(labels).tolist()
    }

# %%
def cluster_by_genre_parallel(filepath: str, output_dir: str = "models", max_workers: int = None) -> Dict[str, Dict]:
    """
    Main function to cluster music data by genre and save the models using parallel processing
    
    Parameters:
    filepath (str): Path to the CSV file containing music features
    output_dir (str): Directory to save models
    max_workers (int): Maximum number of threads to use (None = auto-determine)
    
    Returns:
    dict: Results dictionary containing clustering results for each genre
    """
    start_time = time.time()
    
    # Load data
    df = load_and_prepare_data(filepath)
    
    # Split by genre
    genre_dfs = split_data_by_genre(df)
    
    results = {}
    
    # Process genres in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dictionary of future: genre pairs
        future_to_genre = {
            executor.submit(process_genre, genre, genre_df, output_dir): genre
            for genre, genre_df in genre_dfs.items()
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_genre):
            genre = future_to_genre[future]
            try:
                result = future.result()
                results[genre] = result
            except Exception as e:
                print(f"Error processing genre {genre}: {str(e)}")
                results[genre] = {"error": str(e)}
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Add execution stats to results
    results["_execution_stats"] = {
        "total_time_seconds": execution_time,
        "num_genres": len(genre_dfs),
        "threads_used": max_workers if max_workers else "auto"
    }
    
    return results

# %%
def cluster_baseline(filepath: str, output_dir: str = "models", max_workers: int = None) -> Dict[str, Dict]:
    """
    Main function to cluster music data by genre and save the models using parallel processing
    
    Parameters:
    filepath (str): Path to the CSV file containing music features
    output_dir (str): Directory to save models
    max_workers (int): Maximum number of threads to use (None = auto-determine)
    
    Returns:
    dict: Results dictionary containing clustering results for each genre
    """
    start_time = time.time()
    
    # Load data
    df = load_and_prepare_data(filepath)

    results = {}
    
    # Process genres in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a dictionary of future: genre pairs
        genre = 'Baseline'
        genre_df = df
        future = executor.submit(process_genre, genre, genre_df, output_dir)
        
        # Process results as they complete
        try:
            results['Baseline'] = future.result()
        except Exception as e:
            print(f"Error processing baseline: {str(e)}")
            results = {"error": str(e)}
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Add execution stats to results
    results["_execution_stats"] = {
        "total_time_seconds": execution_time,
        "threads_used": max_workers if max_workers else "auto"
    }
    
    return results


# %%
def predict_with_genre_models(new_data: pd.DataFrame, model_dir: str = "models") -> pd.DataFrame:
    """
    Predict clusters for new data using saved models by genre
    
    Parameters:
    new_data (pd.DataFrame): New music data to cluster
    model_dir (str): Directory containing saved models
    
    Returns:
    pd.DataFrame: Original DataFrame with cluster assignments added
    """
    genre_dfs = split_data_by_genre(new_data)
    result_df = pd.DataFrame()
    
    def predict_genre(genre, genre_df):
        """Helper function for parallel prediction"""
        try:
            # Load the compressed model for this genre
            model = load_compressed_model(genre, model_dir)
            
            # Prepare features
            X = genre_df[['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 
                         'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
                         'valence', 'tempo', 'time_signature', 'encoded_genre']].dropna()
            
            # Get row indices before prediction
            indices = X.index
            
            # Predict clusters
            labels = model.predict(X)
            
            # Add cluster labels to a copy of the original data
            temp_df = genre_df.copy()
            temp_df.loc[indices, 'cluster'] = labels
            
            return temp_df
            
        except Exception as e:
            print(f"Error processing genre {genre}: {str(e)}")
            # Return the data without cluster labels
            return genre_df
    
    # Process predictions in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit prediction tasks
        future_to_genre = {
            executor.submit(predict_genre, genre, genre_df): genre
            for genre, genre_df in genre_dfs.items()
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_genre):
            genre = future_to_genre[future]
            try:
                genre_result = future.result()
                result_df = pd.concat([result_df, genre_result])
            except Exception as e:
                print(f"Error collecting results for genre {genre}: {str(e)}")
    
    return result_df

# %%
def analyze_model_storage(output_dir: str = "models") -> Dict:
    """
    Analyze storage efficiency of compressed models
    
    Parameters:
    output_dir (str): Directory containing saved models
    
    Returns:
    dict: Analysis results
    """
    results = {
        "total_size_kb": 0,
        "num_models": 0,
        "models": []
    }
    
    if not os.path.exists(output_dir):
        return results
    
    for filename in os.listdir(output_dir):
        if filename.endswith('.pkl.gz'):
            filepath = os.path.join(output_dir, filename)
            size_kb = os.path.getsize(filepath) / 1024
            
            # Extract genre from filename
            genre = filename.replace('cluster_model_', '').replace('.pkl.gz', '')
            
            results["models"].append({
                "genre": genre,
                "size_kb": size_kb,
                "path": filepath
            })
            
            results["total_size_kb"] += size_kb
            results["num_models"] += 1
    
    results["avg_size_kb"] = results["total_size_kb"] / results["num_models"] if results["num_models"] > 0 else 0
    
    return results

    # %%
def main():
    # Example usage
    MUSIC_FEATURES_FP = 'data_music_features/processed_spotify_sample.csv'
    # Get today's date
    today_date = datetime.today().strftime('%Y-%m-%d')
    
    # Define output directory with today's date
    OUTPUT_DIR = os.path.join("models", today_date, "genre_cluster_models")

    # Run clustering for all genres in parallel and save compressed models
    results = cluster_by_genre_parallel(MUSIC_FEATURES_FP, output_dir=OUTPUT_DIR)

    # Print execution stats
    stats = results.pop("_execution_stats", {})
    print("\nExecution Statistics:")
    print(f"- Total execution time: {stats.get('total_time_seconds', 0):.2f} seconds")
    print(f"- Number of genres processed: {stats.get('num_genres', 0)}")
    print(f"- Threads used: {stats.get('threads_used', 'unknown')}")

    # Print summary of results
    print("\nClustering Results Summary:")
    for genre, info in results.items():
        if "error" in info:
            print(f"Genre: {genre} - Error: {info['error']}")
            continue
            
        print(f"Genre: {genre}")
        print(f"  - Data shape: {info['data_shape']}")
        print(f"  - Number of clusters: {info['num_clusters']}")
        print(f"  - Model saved to: {info['model_path']}")
        print(f"  - Model size: {info['model_size_kb']:.2f} KB")
        
        # Show top 3 largest clusters if available
        if 'cluster_distribution' in info:
            cluster_dist = np.array(info['cluster_distribution'])
            top_clusters = cluster_dist.argsort()[-3:][::-1]
            print(f"  - Top 3 largest clusters: {top_clusters}")
        print()

    # Analyze storage efficiency
    storage_analysis = analyze_model_storage(OUTPUT_DIR)
    print("\nStorage Analysis:")
    print(f"- Total models: {storage_analysis['num_models']}")
    print(f"- Total storage used: {storage_analysis['total_size_kb']:.2f} KB")
    print(f"- Average model size: {storage_analysis['avg_size_kb']:.2f} KB")


# # %%
# # Example of loading models and predicting new data
# print("\nLoading compressed models to predict on new data...")
# new_df = load_and_prepare_data(MUSIC_FEATURES_FP)  # In practice, this would be new data
# clustered_df = predict_with_genre_models(new_df, model_dir=OUTPUT_DIR)
# print(f"Prediction complete. Output shape: {clustered_df.shape}")


