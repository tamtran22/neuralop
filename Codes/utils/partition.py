import numpy as np
from collections import deque

def bfs_partition_graph(node_features, edges, root, max_partition_size):
    """
    Partitions a graph using BFS while ensuring each partition consists of connected nodes and does not exceed max_partition_size.
    
    :param node_features: List of node features (not used in partitioning, but can be included if needed).
    :param edges: List of tuples representing edges (u, v) in an undirected graph.
    :param root: The starting node for BFS.
    :param max_partition_size: Maximum number of nodes in a partition.
    :return: List of partitions, where each partition is a set of connected nodes.
    """
    from collections import defaultdict
    
    # Build adjacency list
    graph = defaultdict(set)
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    
    visited = set()
    partitions = []
    
    def bfs(start_node):
        queue = deque([start_node])
        partition = set()
        
        while queue and len(partition) < max_partition_size:
            node = queue.popleft()
            if node in visited:
                continue
            
            visited.add(node)
            partition.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor)
        
        return partition
    
    # Perform BFS starting from the root and continue for unvisited nodes
    for n in range(len(node_features)):
        if n not in visited:
            partition = bfs(n)
            if partition:
                partitions.append(partition)
    
    return partitions

if __name__ == "__main__":
    nodes = np.zeros(shape=(11, 3))
    edges = [[0,1],[1,2],[1,3],[2,4],[2,5],[3,6],[3,7],[4,8],[4,9],[5,10]]
    bfs_partition_graph(nodes, edges, root=0, max_partition_size=5)