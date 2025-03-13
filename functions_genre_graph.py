# %%

import networkx as nx
from pyvis.network import Network
import random
import os
import datetime

# %%

def create_music_genre_graph(save_path="music_genre_graph.graphml"):
    """
    Creates a music genre knowledge graph and saves it to a file.
    
    Args:
        save_path (str): Path to save the graph file (default: "music_genre_graph.graphml")
    
    Returns:
        networkx.Graph: The created music genre graph
    """

    # Initialize graph
    G = nx.Graph()
    
    # Define genre groups
    genre_groups = {
        "blues": ['jazz', 'gospel'],
        "country": ['honky-tonk', 'bluegrass', 'folk'],
        "r-n-b": ['groove', 'soul', 'funk', 'disco'],
        "rock": ['rock-n-roll', 'rockabilly', 'j-rock', 'hard-rock', 'psych-rock', 'punk-rock', 'punk',
                'alt-rock', 'alternative', 'grunge', 'emo', 'indie', 'indie-pop', 'hardcore', 'metal', 'black-metal',
                'death-metal', 'heavy-metal', 'metalcore', 'grindcore', 'goth'],
        "pop": ['power-pop', 'synth-pop', 'j-pop', 'j-idol', 'k-pop', 'mandopop',
                'cantopop', 'mpb', 'pop-film'],
        'afrobeat': [],
        "reggae": ['dancehall', 'dub', 'ska', 'latin', 'latino'],
        "hip-hop": [],
        "breakbeat": ['drum-and-bass', 'trip-hop'],
        'house': ['deep-house', 'chicago-house', 'progressive-house'],
        'techno': ['detroit-techno', 'minimal-techno'],
        "edm": ['techno', 'detroit-techno', 'minimal-techno', 'electro', 'electronic', 'dubstep',
                    'idm', 'club', 'hardstyle', 'trance', 'j-dance', 'party'],
        "ambient": ['chill', 'study'],
        "industrial": [],
        'anime': [],
        "classical": ['piano', 'opera', 'romance'],
        'brazil': ['pagode', 'sertanejo', 'forro'],
        'world-music': ['tango', 'samba', 'salsa', 'brazil'],
        "children": ['kids', 'disney', 'show-tunes']
    }
    
    # Define edges
    edges = [
        ('country', 'honky-tonk', 'subgenre'), ('country', 'bluegrass', 'subgenre'), ('bluegrass', 'folk', 'subgenre'),
        ('country', 'rock-n-roll', 'influence'),
        ('r-n-b', 'blues', 'influence'), ('r-n-b', 'soul', 'influence'),
        ('r-n-b', 'jazz', 'influence'), ('funk', 'soul', 'influence'), ('funk', 'disco', 'influence'),
        ('funk', 'psych-rock', 'influence'), ('funk', 'jazz', 'influence'), ('disco', 'synth-pop', 'influence'),
        ('disco', 'soul', 'influence'), ('disco', 'chicago-house', 'influence'), ('groove', 'funk', 'influence'),
        ('groove', 'soul', 'influence'), ('groove', 'jazz', 'influence'), 
        ('rock-n-roll', 'rockabilly', 'subgenre'), ('rockabilly', 'country', 'influence'),
        ('punk-rock', 'punk', 'subgenre'), ('punk-rock', 'grunge', 'influence'), ('punk-rock', 'metal', 'influence'),
        ('punk-rock', 'hardcore', 'influence'), ('punk', 'heavy-metal', 'influence'), ('punk', 'metalcore', 'influence'), 
        ('hardcore', 'heavy-metal', 'influence'), ('hardcore', 'metalcore', 'influence'), ('hardcore', 'punk-rock', 'influence'), 
        ('alt-rock', 'indie-pop', 'influence'),
        ('alt-rock', 'indie', 'influence'), ('alt-rock', 'grunge', 'influence'), ('alternative', 'indie-pop', 'influence'),
        ('alternative', 'indie', 'influence'), ('alternative', 'grunge', 'influence'), ('indie', 'indie-pop', 'influence'),
        ('alt-rock', 'alternative', 'influence'), ('alternative', 'indie', 'influence'), ('emo', 'hardcore', 'influence'),
        ('emo', 'punk', 'influence'), ('emo', 'indie', 'influence'), ('goth', 'punk-rock', 'influence'),
        ('goth', 'industrial', 'influence'), ('grunge', 'punk-rock', 'influence'), ('grunge', 'metal', 'influence'),
        ('metal', 'psych-rock', 'influence'), ('psych-rock', 'jazz', 'influence'), ('psych-rock', 'funk', 'influence'),
        ('synth-pop', 'dance', 'influence'), ('synth-pop', 'pop', 'influence'),
        ('indie-pop', 'pop', 'influence'), ('reggae', 'dancehall', 'subgenre'),
        ('reggae', 'ska', 'influence'), ('reggae', 'dub', 'influence'), ('dancehall', 'ska', 'influence'),
        ('dancehall', 'latin', 'influence'), ('dancehall', 'latino', 'influence'), ('ska', 'soul', 'influence'),
        ('ska', 'r-n-b', 'influence'), ('hip-hop', 'trip-hop', 'influence'), ('hip-hop', 'breakbeat', 'influence'),
        ('breakbeat', 'dubstep', 'influence'), ('breakbeat', 'trip-hop', 'influence'), ('drum-and-bass', 'techno', 'influence'),
        ('dubstep', 'edm', 'influence'), ('detroit-techno', 'chicago-house', 'influence'),
        ('detroit-techno', 'trance', 'influence'), ('deep-house', 'chicago-house', 'influence'),
        ('trance', 'ambient', 'influence'), ('anime', 'j-pop', 'influence'),
        ('anime', 'j-idol', 'influence'), ('gospel', 'r-n-b', 'derivative'), 
        ('gospel', 'soul', 'derivative'), ('study', 'chill', 'connect'), 
        ('show-tunes', 'pop-film', 'connect'), ('show-tunes', 'disney', 'connect'),
        ("rockabilly", "rock-n-roll", "subgenre"), ("rockabilly", "country", "influence"),
        ("punk-rock", "punk", "subgenre"), ("punk-rock", "grunge", "influence"), ("punk-rock", "metal", "influence"),
        ("punk-rock", "hardcore", "influence"), ("punk", "heavy-metal", "influence"), ("punk", "metalcore", "influence"),
        ("hardcore", "heavy-metal", "influence"), ("hardcore", "metalcore", "influence"),
        ("hardcore", "punk-rock", "influence"),
        ("alt-rock", "indie-pop", "influence"), ("alt-rock", "indie", "influence"), ("alt-rock", "grunge", "influence"),
        ("alternative", "indie-pop", "influence"), ("alternative", "indie", "influence"), ("alternative", "grunge", "influence"),
        ("indie", "indie-pop", "influence"), ("alt-rock", "alternative", "influence"), ("alternative", "indie", "influence"),
        ("emo", "hardcore", "influence"), ("emo", "punk", "influence"), ("emo", "indie-pop", "influence"),
        ("goth", "punk-rock", "influence"), ("goth", "industrial", "influence"),
        ("grunge", "punk-rock", "influence"), ("grunge", "metal", "influence"),
        ("grindcore", "heavy-metal", "influence"), ("metal", "punk-rock", "influence"), ("metal", "psych-rock", "influence"),
        ("black-metal", "heavy-metal", "influence"), ("death-metal", "metalcore", "influence"), ("death-metal", "heavy-metal", "influence"),
        ("heavy-metal", "black-metal", "influence"), ("heavy-metal", "death-metal", "influence"), ("heavy-metal", "grindcore", "influence"),
        ("heavy-metal", "hardcore", "influence"), ("heavy-metal", "punk", "influence"),
        ("metalcore", "hardcore", "influence"), ("metalcore", "death-metal", "influence"), ("metalcore", "punk", "influence"),
        ("psych-rock", "goth", "influence"), ("psych-rock", "metal", "influence"), ("psych-rock", "jazz", "influence"), ("psych-rock", "funk", "influence"),
        ('k-pop', 'j-pop', 'influence'), ("idm", "detroit-techno", "influence"), ("idm", "techno", "influence"), ("idm", "breakbeat", "influence"),
        ("hardstyle", "trance", "influence"), ("hardstyle", "techno", "influence"), ("hardstyle", "house", "influence"),
        ("trance", "house", "influence"), ("trance", "ambient", "influence"), ("trance", "detroit-techno", "influence"),
        ("j-dance", "j-pop", "influence"), ("j-dance", "house", "influence"), ("j-dance", "techno", "influence"), ("j-dance", "trance", "influence"),
        ("electro", "house", "influence"), ("electro", "techno", "influence"), ("electro", "edm", "influence"),
        ("breakbeat", "dubstep", "influence"), ("breakbeat", "trip-hop", "influence"), ("drum-and-bass", "techno", "influence"),
        ("dubstep", "edm", "influence"), ("detroit-techno", "chicago-house", "influence"), ("detroit-techno", "trance", "influence"),
        ("deep-house", "chicago-house", "influence"), ("techno", "detroit-techno", "influence"),
        ("techno", "minimal-techno", "influence"), ("detroit-techno", "minimal-techno", "influence"), ("detroit-techno", "electro", "influence"),
        ("chicago-house", "disco", "influence"), ("chicago-house", "ambient", "influence"), ('dance', 'edm', 'influence'),
        ('dance', 'hip-hop', 'influence'), ('dance', 'disco', 'influence'), ('dance', 'dancehall', 'influence'),
        ('industrial', 'techno', 'influence'), ('indie-pop', 'power-pop', 'influence'), ('mandopop', 'cantopop', 'influence'),
        ('hardcore', 'punk', 'influence'), ('brazil', 'mpb', 'influence'), ('brazil', 'pagode', 'influence'), 
        ('afrobeat', 'jazz', 'influence'), ('afrobeat', 'funk', 'influence'), ('afrobeat', 'soul', 'influence'), 
        ('forro', 'samba', 'influence'), ('forro', 'tango', 'influence'), ('forro', 'salsa', 'influence'), 
        ('salsa', 'samba', 'influence'), ('tango', 'samba', 'influence'), ('salsa', 'tango', 'influence'), ('salsa', 'latin', 'influence'),
        ('pagode', 'samba', 'influence'), ('party', 'club', 'influence'), ('j-dance', 'dance', 'influence')
    ]
    
    # Create Graph
    for group, subgenres in genre_groups.items():
        for genre in subgenres:
            G.add_node(genre, group=group)
            G.add_edge(group, genre, label='subgenre')
    for edge in edges:
        G.add_edge(edge[0], edge[1], label=edge[2])
    
    # Save the graph to a file in todays folder
    today = datetime.datetime.now().strftime("%y-%m-%d")
    save_dir = os.path.join("models", today)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_path)
    nx.write_graphml(G, save_path)
    print(f"Graph saved to {save_path}")
    
    return G

# %%

def visualize_graph(G, save_path):
    # Optional: Create and save visualization
    net = Network(notebook=True, height='750px', width='100%', bgcolor='#222222', font_color='white')
    net.from_nx(G)

    today = datetime.datetime.now().strftime("%y-%m-%d")
    save_dir = os.path.join("models", today)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_path)
    
    html_path = save_path.replace('.graphml', '.html') if save_path.endswith('.graphml') else None #save_path + '.html'
    net.show(html_path)
    print(f"Interactive visualization saved to {html_path}")

# %%

def get_next_genre_limit(G, current_genre, original_genre, distance_penalty=3):
    """
    Determine the next genre to transition to from the current genre based on shortest path lengths and a distance penalty.
    Parameters:
    G (networkx.Graph): The graph representing genres and their connections.
    current_genre (str): The current genre from which to determine the next genre.
    original_genre (str): The original genre to compute shortest paths from.
    distance_penalty (int, optional): The penalty factor applied to the distance from the original genre. Default is 3.
    Returns:
    str: The next genre to transition to, selected based on weighted probabilities.
    """

    # Compute shortest paths from the original genre (country) to all genres
    original_shortest_paths = nx.single_source_shortest_path_length(G, source=original_genre)
    
    # Compute shortest paths from the current genre
    shortest_paths = nx.single_source_shortest_path(G, source=current_genre)

    # Extract all possible next genres
    next_genre_candidates = {}
    for target, path in shortest_paths.items():
        if len(path) > 1:  # Ignore self-paths
            next_genre = path[1]  # First step in the path
            path_length = len(path) - 1  # Exclude the source
            if next_genre not in next_genre_candidates:
                next_genre_candidates[next_genre] = []
            next_genre_candidates[next_genre].append(path_length)
    
    # Compute weighted probabilities (favor shorter paths)
    genre_weights = {}
    for genre, path_lengths in next_genre_candidates.items():
        avg_path_length = sum(path_lengths) / len(path_lengths)  # Average path length to target
        weight = 1 / (avg_path_length + 1)  # Shorter paths get higher probability

        # Compute distance from original genre
        distance_from_original = original_shortest_paths.get(genre, float('inf'))

        # Apply a distance penalty: If far from original genre, reduce probability
        distance_weight = 1 / (distance_from_original ** distance_penalty + 1)
        
        # Final weight
        genre_weights[genre] = weight * distance_weight

    # Normalize weights to form a probability distribution
    total_weight = sum(genre_weights.values())
    genre_probabilities = {genre: weight / total_weight for genre, weight in genre_weights.items()}

    # Select next genre randomly based on computed probabilities
    next_genre = random.choices(list(genre_probabilities.keys()), weights=genre_probabilities.values(), k=1)[0]

    return next_genre

# %%

def get_transitions(G, original_genre, n, diversity_penalty=3):
    """
    Generate a sequence of genres starting from the original genre.
    Generates list of n+1 genres b/c first genre is known.
    Parameters:
    G (networkx.Graph): The graph representing the genre transitions.
    original_genre (str): The starting genre.
    n (int): The number of genres to generate in the sequence.
    diversity_penalty (int, optional): The penalty for revisiting genres. Default is 3.
    Returns:
    list: A list of genres representing the transition sequence.
    """
    
    genres = [original_genre]
    nxt = get_next_genre_limit(G, original_genre, original_genre, diversity_penalty)
    genres.append(nxt)
    for i in range(n-1):
        nxt = get_next_genre_limit(G, nxt, original_genre, diversity_penalty)
        genres.append(nxt)
        
    return genres
