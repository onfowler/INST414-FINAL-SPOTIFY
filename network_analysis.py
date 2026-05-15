import pandas as pd
import networkx as nx
import re

print("Loading dataset and scanning for collaborations...")
df = pd.read_csv('songs_normalize.csv')

# Initialize an empty graph
G = nx.Graph()

# Function to extract the featured artist's name from the song title
def get_featured_artist(song_title):
    # Searches for "feat." or "feat " and grabs the name inside the parentheses 
    match = re.search(r'feat\.?\s*([^)]+)', str(song_title), re.IGNORECASE)
    if match:
        # Clean up the string to grab just the primary featured artist
        name = match.group(1).split('&')[0].split(',')[0].strip(' )]')
        return name
    return None

# Build the network
for index, row in df.iterrows():
    primary_artist = row['artist'].strip()
    feat_artist = get_featured_artist(row['song'])
    
    if feat_artist:
        # Draw a connection (edge) between the two artists (nodes)
        G.add_edge(primary_artist, feat_artist)

print(f"\nNetwork successfully built!")
print(f"Total Artists (Nodes): {G.number_of_nodes()}")
print(f"Total Collaborations (Edges): {G.number_of_edges()}")

# Calculate Centrality Metrics
degree_dict = nx.degree_centrality(G)
betweenness_dict = nx.betweenness_centrality(G)

# Sort the dictionaries to find the top 5 highest scores
top_degree = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
top_betweenness = sorted(betweenness_dict.items(), key=lambda x: x[1], reverse=True)[:5]

print("\n--- TOP 5 ARTISTS: DEGREE CENTRALITY (Most Collaborations) ---")
for artist, score in top_degree:
    print(f"{artist}: {score:.4f}")

print("\n--- TOP 5 ARTISTS: BETWEENNESS CENTRALITY (Biggest Network Bridges) ---")
for artist, score in top_betweenness:
    print(f"{artist}: {score:.4f}")