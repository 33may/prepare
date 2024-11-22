import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Reinitialize Node class for the graph structure
class Node:
    def __init__(self, layer, layer_down=None, neighbours=None, data=None):
        self.layer = layer
        self.layer_down = layer_down
        self.neighbours = neighbours if neighbours else []
        self.data = data

# Create nodes with data (coordinates)
node1 = Node(layer=0, data=np.array([0, 0]))
node2 = Node(layer=0, data=np.array([0, 3]))
node3 = Node(layer=0, data=np.array([2, 2]))
node4 = Node(layer=0, data=np.array([2, 5]))
node5 = Node(layer=0, data=np.array([4, 7]))
node6 = Node(layer=0, data=np.array([2, 0]))

# Define neighbors for each node
node1.neighbours = [node2, node6, node4]
node2.neighbours = [node3, node1]
node3.neighbours = [node2, node6, node5]
node4.neighbours = [node1, node5]
node5.neighbours = [node3, node4, node6]
node6.neighbours = [node1, node5, node3]

# Initial entry point for the search
ep = node5

# Compute Euclidean distance
def compute_distance(input, query):
    return np.linalg.norm(input - query)

# Create a graph for visualization
G = nx.Graph()

# Add nodes with coordinates as attributes
nodes = [node1, node2, node3, node4, node5, node6]
for i, node in enumerate(nodes):
    G.add_node(i, pos=node.data)

# Add edges based on neighbors
for i, node in enumerate(nodes):
    for neighbor in node.neighbours:
        G.add_edge(i, nodes.index(neighbor))

# Define positions for nodes
positions = nx.get_node_attributes(G, 'pos')

# Search algorithm with recording for visualization
visited_nodes = []
visited_edges = []

def search_layer(q, ep, ef):
    """
    Implements SEARCH-LAYER with synchronization for visualization.
    """
    visited = [ep]
    candidates = [(ep, compute_distance(ep.data, q))]
    result = [(ep, compute_distance(ep.data, q))]
    visited_nodes.clear()
    visited_edges.clear()

    while len(candidates) > 0:
        # Find the closest candidate
        candidate = min(candidates, key=lambda x: x[1])
        candidates.remove(candidate)

        # Synchronize node highlighting
        if nodes.index(candidate[0]) not in visited_nodes:
            visited_nodes.append(nodes.index(candidate[0]))

        # Find the furthest element in the result
        farthest = max(result, key=lambda x: x[1])

        if candidate[1] > farthest[1]:
            break
        else:
            neighbors = candidate[0].neighbours
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.append(neighbor)
                    neighbor_tuple = (neighbor, compute_distance(neighbor.data, q))
                    farthest = max(result, key=lambda x: x[1])

                    if (neighbor_tuple[1] < farthest[1]) or (len(result) < ef):
                        candidates.append(neighbor_tuple)
                        result.append(neighbor_tuple)

                        # Synchronize edge highlighting
                        visited_edges.append((nodes.index(candidate[0]), nodes.index(neighbor)))

                        if len(result) > ef:
                            result.remove(max(result, key=lambda x: x[1]))
    return result



# Perform search and record visited nodes/edges
search_layer(q=np.array([0, 1]), ep=ep, ef=2)

# Visualization setup
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 8)
ax.set_title("Graph Search Visualization")

def init():
    nx.draw_networkx_edges(G, pos=positions, ax=ax, alpha=0.3, edge_color="gray")
    nx.draw_networkx_nodes(G, pos=positions, ax=ax, node_color="blue", alpha=0.6)
    return ax,

def update(frame):
    ax.clear()
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 8)
    ax.set_title("Graph Search Visualization")

    # Highlight visited nodes and edges
    current_nodes = visited_nodes[:frame+1]
    current_edges = visited_edges[:frame]

    # Highlight edges and nodes visited up to current frame
    nx.draw_networkx_edges(G, pos=positions, ax=ax, edgelist=current_edges, edge_color="red", width=2)
    nx.draw_networkx_nodes(G, pos=positions, nodelist=current_nodes, node_color="red", ax=ax)

    # Draw full graph as background
    nx.draw_networkx_edges(G, pos=positions, ax=ax, alpha=0.3, edge_color="gray")
    nx.draw_networkx_nodes(G, pos=positions, node_color="blue", ax=ax, alpha=0.6)
    nx.draw_networkx_labels(G, pos=positions, ax=ax, font_size=10)

    return ax,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(visited_nodes), init_func=init, blit=False, interval=1000)

# Save the animation as an MP4 video
video_path = "./graph_search_animation.mp4"
writer = FFMpegWriter(fps=2, metadata=dict(artist='Me'), bitrate=1800)
ani.save(video_path, writer=writer)

video_path
