# %%
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.spatial.distance import cdist
import warnings

# %%
warnings.simplefilter(action='ignore', category=UserWarning)

# %%
def prepare_features_for_clustering(
    df: pd.DataFrame, 
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare feature data for clustering by removing rows with NaN values.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing song data
    feature_cols : list of str, optional
        Feature columns to use. If None, uses default features
        
    Returns:
    --------
    tuple
        (Clean DataFrame with all columns, DataFrame with only features)
    """
    if not feature_cols:
        feature_cols = ['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
                        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                        'valence', 'tempo', 'time_signature', 'encoded_genre']
    
    # Extract feature columns
    df_features = df[feature_cols].copy()
    
    # Remove rows with NaN values
    df_features = df_features.dropna()
    
    # Filter main dataframe to same rows
    clean_df = df.loc[df_features.index]
    
    return clean_df, df_features

# %%
def select_starting_song(
    df: pd.DataFrame, 
    starting_song_index: Optional[int] = None,
) -> int:
    """
    Select a starting song index, either from input or randomly.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing songs
    starting_song_index : int, optional
        Index of starting song. If None or invalid, selects randomly
        
    Returns:
    --------
    int
        Index of the starting song
    """
    if starting_song_index is None or starting_song_index not in df.index:
        return np.random.choice(df.index)
    return starting_song_index

# %%
def get_cluster_assignments(
    model, 
    df_features: pd.DataFrame
) -> np.ndarray:
    """
    Get cluster assignments for all songs in the dataset.
    
    Parameters:
    -----------
    model : clustering model
        Model with predict() method
    df_features : pandas DataFrame
        DataFrame containing song features
        
    Returns:
    --------
    numpy.ndarray
        Array of cluster assignments
    """
    return model.predict(df_features)

# %%
def compute_cluster_transition_probabilities(
    model, 
    current_cluster: int, 
    temperature: float
) -> np.ndarray:
    """
    Compute probabilities for transitioning to other clusters.
    
    Parameters:
    -----------
    model : clustering model
        Model with cluster_centers_ attribute
    current_cluster : int
        Current cluster ID
    temperature : float
        Controls randomness (lower = more predictable)
        
    Returns:
    --------
    numpy.ndarray
        Array of probabilities for each cluster
    """
    # if not hasattr(model, 'cluster_centers_'):
    #     # If model doesn't have centroids, return None
    #     return None
        
    centroids = model.cluster_centers_
    # Calculate distances from current cluster to all clusters
    distances = cdist([centroids[current_cluster]], centroids)[0]
    
    # Convert distances to probabilities (smaller distance = higher probability)
    # Apply temperature to control randomness
    probs = np.exp(-distances / temperature)
    return probs / probs.sum()

# %%
def select_next_cluster(
    model,
    current_cluster: int,
    unique_clusters: np.ndarray,
    temperature: float,
    explore: bool = False
) -> int:
    """
    Select the next cluster to sample from.
    
    Parameters:
    -----------
    model : clustering model
        Clustering model with predict() method
    current_cluster : int
        Current cluster ID
    unique_clusters : numpy.ndarray
        Array of unique cluster IDs
    temperature : float
        Controls randomness
    explore : bool
        Whether to force exploration of other clusters
        
    Returns:
    --------
    int
        Next cluster ID
    """
    # If not exploring, stay in current cluster
    if not explore:
        return current_cluster
        
    # # When exploring, sample from clusters based on distance
    # if hasattr(model, 'cluster_centers_'):
    # Get transition probabilities
    all_probs = compute_cluster_transition_probabilities(
        model, current_cluster, temperature
    )

    # Extract probabilities only for unique clusters we're considering
    # This is the critical fix - ensuring probs matches unique_clusters
    probs = np.zeros(len(unique_clusters))
    for i, cluster_id in enumerate(unique_clusters):
        # Make sure cluster_id is within range of all_probs
        if cluster_id < len(all_probs):
            probs[i] = all_probs[cluster_id]

    # Normalize to ensure probabilities sum to 1
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    else:
        # Fallback to uniform distribution if all probs are zero
        probs = np.ones(len(unique_clusters)) / len(unique_clusters)

    # Check for NaN values which would cause another error
    if np.any(np.isnan(probs)):
        # Replace NaNs with uniform distribution
        probs = np.ones(len(unique_clusters)) / len(unique_clusters)

    #print(probs)
    # Sample a cluster based on proximity
    return np.random.choice(unique_clusters, p=probs)
    # else:
    #     # Simple random selection if no centroids available
    #     other_clusters = [c for c in unique_clusters if c != current_cluster]
    #     if other_clusters:
    #         return np.random.choice(other_clusters)
    #     return current_cluster

# %%
def get_available_songs(
    df: pd.DataFrame,
    all_clusters: np.ndarray,
    cluster: int,
    selected_indices: set
) -> List[int]:
    """
    Get available (not yet selected) songs from a cluster.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing songs
    all_clusters : numpy.ndarray
        Array of cluster assignments for all songs
    cluster : int
        Cluster ID to find songs from
    selected_indices : set
        Set of already selected song indices
        
    Returns:
    --------
    list
        List of available song indices
    """
    cluster_songs = df.index[all_clusters == cluster]
    return [idx for idx in cluster_songs if idx not in selected_indices]

# %%
def find_cluster_with_available_songs(
    df: pd.DataFrame,
    all_clusters: np.ndarray,
    unique_clusters: np.ndarray,
    selected_indices: set
) -> Optional[int]:
    """
    Find a cluster that still has available songs.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing songs
    all_clusters : numpy.ndarray
        Array of cluster assignments for all songs
    unique_clusters : numpy.ndarray
        Array of unique cluster IDs
    selected_indices : set
        Set of already selected song indices
        
    Returns:
    --------
    int or None
        Cluster ID with available songs, or None if all songs are selected
    """
    for cluster in unique_clusters:
        available_songs = get_available_songs(
            df, all_clusters, cluster, selected_indices
        )
        if available_songs:
            return cluster
    return None

# %%
def extract_human_readable_features(song: Dict) -> Dict:
    """
    Extract only human-readable features from a song dictionary.
    
    Parameters:
    -----------
    song : dict
        Dictionary containing song information
        
    Returns:
    --------
    dict
        Dictionary with only human-readable features
    """
    # Define human-readable features to keep
    readable_fields = [
        'artists','album_name','track_name','track_genre'
    ]
    
    # Create a new dictionary with only the readable fields
    readable_song = {}
    for field in readable_fields:
        if field in song and pd.notna(song[field]):
            readable_song[field] = song[field]
    
    return readable_song

# %%
def generate_stochastic_playlist(
    df: pd.DataFrame, 
    model, 
    starting_song_index: Optional[int] = None, 
    playlist_size: int = 20, 
    temperature: float = 0.3, 
    feature_cols: Optional[List[str]] = None,
    exploration_rate: float = 0.3  # 30% chance to explore
) -> List[Dict]:
    """
    Generate a playlist by stochastically sampling songs from clusters.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing encoded songs with features
    model : object
        Clustering model with predict() method
    starting_song_index : int, optional
        Index of the starting song. If None, a random song is selected
    playlist_size : int
        Number of songs to include in the playlist
    temperature : float
        Controls randomness (0.0 = deterministic, 1.0+ = more random)
    feature_cols : list, optional
        Feature columns to use for prediction
    exploration_rate : float
        Probability of exploring other clusters
        
    Returns:
    --------
    list
        List of dictionaries containing song information for the playlist
    """
    # Prepare data
    df, df_features = prepare_features_for_clustering(df, feature_cols)
    
    # Get cluster assignments for all songs
    all_clusters = get_cluster_assignments(model, df_features)
    df['cluster'] = all_clusters
    unique_clusters = np.unique(all_clusters)
    #print(unique_clusters)
    
    # Select starting song
    starting_song_index = select_starting_song(df, starting_song_index)
    
    # Get starting song features and determine its cluster
    starting_features = df_features.loc[starting_song_index].values.reshape(1, -1)
    current_cluster = model.predict(starting_features)[0]
    
    # Initialize playlist
    playlist = [df.loc[starting_song_index].to_dict()]
    selected_indices = {starting_song_index}
    
    # Generate the rest of the playlist
    for _ in range(playlist_size - 1):
        # Decide whether to explore other clusters
        explore = np.random.random() < exploration_rate
        
        # Select next cluster (either stay or explore)
        next_cluster = select_next_cluster(
            model, current_cluster, unique_clusters, temperature, explore
        )
        
        # Get available songs in the selected cluster
        available_songs = get_available_songs(
            df, all_clusters, next_cluster, selected_indices
        )
        
        # If no songs available in that cluster, find another with available songs
        if not available_songs:
            next_cluster = find_cluster_with_available_songs(
                df, all_clusters, unique_clusters, selected_indices
            )
            
            # If no cluster has available songs, we've used all songs
            if next_cluster is None:
                break
                
            # Get available songs from the new cluster
            available_songs = get_available_songs(
                df, all_clusters, next_cluster, selected_indices
            )
        
        # Select a random song from available songs
        next_song_index = np.random.choice(available_songs)
        playlist.append(df.loc[next_song_index].to_dict())
        selected_indices.add(next_song_index)
        
        # Update current cluster for smooth transitions
        next_features = df_features.loc[next_song_index].values.reshape(1, -1)
        current_cluster = model.predict(next_features)[0]
    
    return playlist

# %%
def generate_basic_playlist(
    df: pd.DataFrame, 
    model, 
    starting_song_index: Optional[int] = None, 
    playlist_size: int = 20, 
    #temperature: float = 0.3, 
    feature_cols: Optional[List[str]] = None
    #exploration_rate: float = 0.3  # 30% chance to explore
) -> List[Dict]:
    """
    Generate a playlist by sampling songs from nearest cluster.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing encoded songs with features
    model : object
        Clustering model with predict() method
    starting_song_index : int, optional
        Index of the starting song. If None, a random song is selected
    playlist_size : int
        Number of songs to include in the playlist
    temperature : float
        Controls randomness (0.0 = deterministic, 1.0+ = more random)
    feature_cols : list, optional
        Feature columns to use for prediction
    exploration_rate : float
        Probability of exploring other clusters
        
    Returns:
    --------
    list
        List of dictionaries containing song information for the playlist
    """
    # Prepare data
    df, df_features = prepare_features_for_clustering(df, feature_cols)
    
    # Get cluster assignments for all songs
    all_clusters = get_cluster_assignments(model, df_features)
    df['cluster'] = all_clusters
    unique_clusters = np.unique(all_clusters)
    
    # Select starting song
    starting_song_index = select_starting_song(df, starting_song_index)
    
    # Get starting song features and determine its cluster
    starting_features = df_features.loc[starting_song_index].values.reshape(1, -1)
    current_cluster = model.predict(starting_features)[0]
    
    # Initialize playlist
    playlist = [df.loc[starting_song_index].to_dict()]
    selected_indices = {starting_song_index}
    
    # Generate the rest of the playlist
    for _ in range(playlist_size - 1):
        
        # stay at same cluster
        next_cluster = current_cluster
        
        # Get available songs in the selected cluster
        available_songs = get_available_songs(
            df, all_clusters, next_cluster, selected_indices
        )
        
        # If no songs available in that cluster, find another with available songs
        if not available_songs:
            next_cluster = find_cluster_with_available_songs(
                df, all_clusters, unique_clusters, selected_indices
            )
            
            # If no cluster has available songs, we've used all songs
            if next_cluster is None:
                break
                
            # Get available songs from the new cluster
            available_songs = get_available_songs(
                df, all_clusters, next_cluster, selected_indices
            )
        
        # Select a random song from available songs
        next_song_index = np.random.choice(available_songs)
        playlist.append(df.loc[next_song_index].to_dict())
        selected_indices.add(next_song_index)
        
        # Update current cluster for smooth transitions
        next_features = df_features.loc[next_song_index].values.reshape(1, -1)
        current_cluster = model.predict(next_features)[0]
    
    return playlist

# %%

def generate_genre_guided_playlist(
    df: pd.DataFrame, 
    model, 
    music_graph,
    #starting_genre: str = 'dance',
    starting_song_index: Optional[int] = None, 
    playlist_size: int = 20, 
    diversity_penalty: float = 3,
    feature_cols: Optional[List[str]] = None
) -> List[Dict]:
    """
    Generate a playlist that follows a genre transition path from the music genre knowledge graph.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing encoded songs with features
    model : object
        Clustering model with predict() method
    music_graph : networkx.Graph
        Music genre knowledge graph
    starting_genre : str
        The genre to start the genre transition path from
    starting_song_index : int, optional
        Index of the starting song. If None, a random song within the starting genre is selected
    playlist_size : int
        Number of songs to include in the playlist
    diversity_penalty : float
        Controls how far genre transitions can go from the original genre
    feature_cols : list, optional
        Feature columns to use for prediction
        
    Returns:
    --------
    list
        List of dictionaries containing song information for the playlist
    """
    from functions.functions_genre_graph import get_transitions

    # Prepare data
    df, df_features = prepare_features_for_clustering(df, feature_cols)
    
    # Get cluster assignments for all songs
    all_clusters = get_cluster_assignments(model, df_features)
    df['cluster'] = all_clusters
    unique_clusters = np.unique(all_clusters)
    
    # Identify which clusters contain songs of each genre
    genre_to_clusters = {}
    for genre in set(df['track_genre']):
        genre_to_clusters[genre] = set(df[df['track_genre'] == genre]['cluster'].unique())
    
    # If starting_song_index is not provided, select starting song
    starting_song_index = select_starting_song(df, starting_song_index)

    # Generate genre transition path
    starting_genre = df.loc[starting_song_index]['track_genre']
    genre_transitions = get_transitions(music_graph, starting_genre, playlist_size, diversity_penalty)
    print(f"Genre transition path: {genre_transitions}")
    
    # Get starting song features and determine its cluster
    starting_features = df_features.loc[starting_song_index].values.reshape(1, -1)
    current_cluster = model.predict(starting_features)[0]
    
    # Initialize playlist
    playlist = [df.loc[starting_song_index].to_dict()]
    selected_indices = {starting_song_index}
    
    # Generate the rest of the playlist based on genre transitions
    for i in range(1, len(genre_transitions)):
        target_genre = genre_transitions[i]
        
        # Strategy 1: First try current cluster with target genre
        target_genre_songs = df[
            (df['track_genre'] == target_genre) & 
            (df['cluster'] == current_cluster) &
            (~df.index.isin(selected_indices))
        ]
        
        # Strategy 2: If no match, find the nearest clusters that contain the target genre
        if target_genre_songs.empty:
            found_song = False
            
            # Find which clusters contain songs of target genre
            if target_genre in genre_to_clusters and genre_to_clusters[target_genre]:
                # Get centroids of all clusters
                centroids = model.cluster_centers_
                # Calculate distances from current cluster to all clusters
                current_centroid = centroids[current_cluster]
                distances_to_clusters = {}
                
                # Calculate distance to each cluster that contains the target genre
                for cluster_id in genre_to_clusters[target_genre]:
                    distance = np.linalg.norm(current_centroid - centroids[cluster_id])
                    distances_to_clusters[cluster_id] = distance
                
                # Sort clusters by proximity to current cluster
                nearest_clusters = sorted(distances_to_clusters.items(), key=lambda x: x[1])
                
                # Try each cluster in order of proximity
                for nearest_cluster_id, _ in nearest_clusters:
                    cluster_genre_songs = df[
                        (df['track_genre'] == target_genre) & 
                        (df['cluster'] == nearest_cluster_id) &
                        (~df.index.isin(selected_indices))
                    ]
                    
                    if not cluster_genre_songs.empty:
                        # Select a random song from the nearest cluster with target genre
                        next_song_index = np.random.choice(cluster_genre_songs.index)
                        found_song = True
                        break
            
            # Strategy 3: If still no match, look for any song with target genre
            if not found_song:
                any_genre_songs = df[
                    (df['track_genre'] == target_genre) & 
                    (~df.index.isin(selected_indices))
                ]
                
                if not any_genre_songs.empty:
                    # Select a random song with target genre from any cluster
                    next_song_index = np.random.choice(any_genre_songs.index)
                    found_song = True
                
                # Strategy 4: If still no match, use current cluster
                if not found_song:
                    print(f"No songs found for genre '{target_genre}', using current cluster.")
                    # Get available songs in the current cluster
                    available_songs = get_available_songs(
                        df, all_clusters, current_cluster, selected_indices
                    )
                    
                    # If no songs available in that cluster, find another with available songs
                    if not available_songs:
                        # Find a new cluster with available songs
                        for cluster_id in sorted(unique_clusters, key=lambda c: np.linalg.norm(centroids[current_cluster] - centroids[c])):
                            available_songs = get_available_songs(
                                df, all_clusters, cluster_id, selected_indices
                            )
                            if available_songs:
                                next_song_index = np.random.choice(available_songs)
                                found_song = True
                                break
                        
                        # If we've exhausted all clusters, we've used all songs
                        if not found_song:
                            print("No more songs available, ending playlist generation.")
                            break
                    else:
                        next_song_index = np.random.choice(available_songs)
        else:
            # Select a random song from available target genre songs in current cluster
            next_song_index = np.random.choice(target_genre_songs.index)
        
        # Add song to playlist
        playlist.append(df.loc[next_song_index].to_dict())
        selected_indices.add(next_song_index)
        
        # Update current cluster for the next iteration
        next_features = df_features.loc[next_song_index].values.reshape(1, -1)
        current_cluster = model.predict(next_features)[0]
        
        # If we've reached the desired playlist size, break
        if len(playlist) >= playlist_size:
            break
    
    return playlist

# %%
def save_playlist_to_csv(playlist: List[Dict], filepath: str) -> None:
    """
    Save the generated playlist to a CSV file with only human-readable features.
    
    Parameters:
    -----------
    playlist : list
        List of dictionaries containing song information
    filepath : str
        Path to save the CSV file
    """
    # Process each song to keep only human-readable features
    readable_playlist = [extract_human_readable_features(song) for song in playlist]
    
    # Convert to DataFrame and save
    playlist_df = pd.DataFrame(readable_playlist)
    playlist_df.to_csv(filepath, index=False)
    print(f"Playlist saved to {filepath}")

# %%
def display_playlist(playlist: List[Dict], show_features: bool = False) -> None:
    """
    Display the generated playlist in a readable format.
    
    Parameters:
    -----------
    playlist : list
        List of dictionaries containing song information
    show_features : bool
        Whether to display audio features
    """
    print(f"\n{'=' * 50}")
    print(f"Generated Playlist ({len(playlist)} songs)")
    print(f"{'=' * 50}")
    
    for i, song in enumerate(playlist, 1):
        # Extract song information (adjust based on your data structure)
        title = song.get('name', song.get('track_name', 'Unknown Title'))
        artist = song.get('artists', song.get('artist_name', 'Unknown Artist'))
        genre = song.get('track_genre', 'Unknown Genre')
        
        print(f"{i}. {title} - {artist} - {genre}")
        
        if show_features and any(key in song for key in ['danceability', 'energy', 'valence']):
            # Filter for only audio features if requested
            features = {k: v for k, v in song.items() if k in [
                'danceability', 'energy', 'valence', 'tempo', 
                'acousticness', 'instrumentalness'
            ]}
            
            # Print formatted features
            if features:
                feature_str = ", ".join([f"{k}: {v:.2f}" for k, v in features.items()])
                print(f"   Features: {feature_str}")
    
    print(f"{'=' * 50}")

# %%
# Example usage
# if __name__ == "__main__":
    # Assuming df and model are already loaded
    # df = pd.read_csv('music_data.csv')
    # model = load_compressed_model('pop', 'models')
    
    # Generate playlist
    # playlist = generate_dynamic_playlist(
    #     df, 
    #     model, 
    #     playlist_size=15, 
    #     temperature=0.4
    # )
    
    # Display the playlist
    # display_playlist(playlist, show_features=True)
    
    # Save to CSV (only human-readable features)
    # save_playlist_to_csv(playlist, 'my_dynamic_playlist.csv')
