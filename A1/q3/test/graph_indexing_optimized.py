# import sys
# import os
# import numpy as np
# from collections import defaultdict, Counter
# import itertools
# import math
# from multiprocessing import Pool, cpu_count
# import hashlib

# #### LLM-assisted code (ChatGPT); all logic and correctness verified by the authors.
# # The DFS-based subgraph enumeration and canonical labeling is adapted from Method 4
# # but modified for graph indexing without class labels.
# ####


# class GraphReader:
#     """Read graphs from dataset files"""
    
#     def read(self, filename):
#         """Read graphs in the given format"""
#         graphs = []
#         current = {'nodes': {}, 'edges': []}
        
#         with open(filename, 'r') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:
#                     if current['nodes'] or current['edges']:
#                         graphs.append(current)
#                         current = {'nodes': {}, 'edges': []}
#                     continue
                
#                 if line.startswith('#'):
#                     if current['nodes'] or current['edges']:
#                         graphs.append(current)
#                     current = {'nodes': {}, 'edges': []}
#                 elif line.startswith('v'):
#                     parts = line.split()
#                     if len(parts) == 3:
#                         nid = int(parts[1])
#                         label = int(parts[2])
#                         current['nodes'][nid] = label
#                 elif line.startswith('e'):
#                     parts = line.split()
#                     if len(parts) == 4:
#                         u = int(parts[1])
#                         v = int(parts[2])
#                         label = int(parts[3])
#                         current['edges'].append((u, v, label))
        
#         if current['nodes'] or current['edges']:
#             graphs.append(current)
        
#         return graphs
    
#     def remove_duplicates(self, graphs):
#         """Remove duplicate graphs while preserving order"""
#         seen = set()
#         unique_graphs = []
        
#         for graph in graphs:
#             # Create canonical representation
#             graph_key = self._graph_to_key(graph)
#             if graph_key not in seen:
#                 seen.add(graph_key)
#                 unique_graphs.append(graph)
        
#         return unique_graphs
    
#     def _graph_to_key(self, graph):
#         """Create unique key for graph"""
#         # Sort nodes by label
#         nodes = sorted((nid, label) for nid, label in graph['nodes'].items())
        
#         # Sort edges with normalized node order
#         edges = []
#         for u, v, label in graph['edges']:
#             u_label = graph['nodes'][u]
#             v_label = graph['nodes'][v]
#             if u_label <= v_label:
#                 edges.append((u_label, v_label, label))
#             else:
#                 edges.append((v_label, u_label, label))
#         edges.sort()
        
#         return f"N{nodes}E{edges}"


# class DFSEnumerator:
#     """DFS-based subgraph enumeration with canonical labeling"""
    
#     def __init__(self, max_nodes=4, min_support=0.05):
#         self.max_nodes = max_nodes
#         self.min_support = min_support
    
#     def enumerate_subgraphs(self, graph):
#         """Enumerate connected subgraphs from a single graph"""
#         subgraphs = set()
        
#         # Convert to adjacency list for faster neighbor access
#         adj_list = self._build_adjacency_list(graph)
#         nodes = list(graph['nodes'].keys())
        
#         # Start DFS from each node
#         for start_node in nodes:
#             self._dfs(start_node, [start_node], [], adj_list, graph, subgraphs)
        
#         return subgraphs
    
#     def _build_adjacency_list(self, graph):
#         """Build adjacency list from graph"""
#         adj = defaultdict(list)
#         for u, v, label in graph['edges']:
#             adj[u].append((v, label))
#             adj[v].append((u, label))
#         return adj
    
#     def _dfs(self, current_node, visited_nodes, visited_edges, adj_list, graph, subgraphs):
#         """Recursive DFS to grow subgraphs"""
#         # Save current subgraph if non-empty
#         if len(visited_nodes) >= 1:
#             subgraph = self._extract_subgraph(visited_nodes, visited_edges, graph)
#             canonical = self._canonical_label(subgraph)
#             subgraphs.add(canonical)
        
#         # Stop if reached max size
#         if len(visited_nodes) >= self.max_nodes:
#             return
        
#         # Explore neighbors
#         for neighbor, edge_label in adj_list[current_node]:
#             # Check if we can add this neighbor
#             if neighbor in visited_nodes:
#                 # Already visited, but could add edge if not already present
#                 edge = (min(current_node, neighbor), max(current_node, neighbor), edge_label)
#                 if edge not in visited_edges:
#                     new_edges = visited_edges + [edge]
#                     self._dfs(current_node, visited_nodes, new_edges, adj_list, graph, subgraphs)
#             else:
#                 # New node
#                 new_nodes = visited_nodes + [neighbor]
#                 edge = (min(current_node, neighbor), max(current_node, neighbor), edge_label)
#                 new_edges = visited_edges + [edge]
#                 self._dfs(neighbor, new_nodes, new_edges, adj_list, graph, subgraphs)
    
#     def _extract_subgraph(self, nodes, edges, graph):
#         """Extract subgraph from node and edge lists"""
#         subgraph_nodes = {}
#         node_mapping = {}
        
#         # Map original node IDs to sequential IDs
#         for i, node_id in enumerate(sorted(nodes)):
#             node_mapping[node_id] = i
#             subgraph_nodes[i] = graph['nodes'][node_id]
        
#         subgraph_edges = []
#         for u, v, label in edges:
#             new_u = node_mapping[min(u, v)]
#             new_v = node_mapping[max(u, v)]
#             subgraph_edges.append((new_u, new_v, label))
        
#         return {'nodes': subgraph_nodes, 'edges': subgraph_edges}
    
#     def _canonical_label(self, subgraph):
#         """Create canonical string representation of subgraph"""
#         # Sort nodes by label, then by degree
#         node_degrees = Counter()
#         for u, v, _ in subgraph['edges']:
#             node_degrees[u] += 1
#             node_degrees[v] += 1
        
#         # Create node tuples (label, degree, original_id) for sorting
#         node_tuples = []
#         for node_id, label in subgraph['nodes'].items():
#             degree = node_degrees[node_id]
#             node_tuples.append((label, degree, node_id))
        
#         # Sort nodes
#         node_tuples.sort()
        
#         # Create mapping to new IDs
#         node_mapping = {}
#         new_nodes = {}
#         for i, (label, degree, old_id) in enumerate(node_tuples):
#             node_mapping[old_id] = i
#             new_nodes[i] = label
        
#         # Apply mapping to edges and sort
#         new_edges = []
#         for u, v, label in subgraph['edges']:
#             new_u = node_mapping[u]
#             new_v = node_mapping[v]
#             if new_u <= new_v:
#                 new_edges.append((new_u, new_v, label))
#             else:
#                 new_edges.append((new_v, new_u, label))
#         new_edges.sort()
        
#         # Create canonical string
#         nodes_str = ','.join(f"{i}:{l}" for i, l in sorted(new_nodes.items()))
#         edges_str = ','.join(f"{u}-{v}:{l}" for u, v, l in new_edges)
        
#         return f"N{nodes_str}|E{edges_str}"


# class PatternSelector:
#     """Select discriminative patterns for graph indexing"""
    
#     def __init__(self, k=50):
#         self.k = k
    
#     def select_patterns(self, pattern_frequencies, database_size):
#         """
#         Select top-k patterns without class labels
#         Uses filtering power instead of class discrimination
#         """
#         scored_patterns = []
        
#         for pattern_str, freq in pattern_frequencies.items():
#             frequency = freq / database_size
            
#             # Skip patterns that are too rare or too common
#             # Ideal for filtering: appears in ~50% of graphs
#             if frequency < 0.05 or frequency > 0.95:
#                 continue
            
#             # Calculate filtering power
#             # Pattern that appears in 50% of graphs can filter 50%
#             filtering_power = 1.0 - abs(0.5 - frequency) * 2
            
#             # Calculate pattern complexity
#             pattern = self._parse_pattern_string(pattern_str)
#             complexity = len(pattern['nodes']) + len(pattern['edges'])
#             complexity_factor = min(complexity, 5) / 5.0
            
#             # Calculate entropy (information content)
#             if frequency > 0 and frequency < 1:
#                 entropy = -frequency * math.log(frequency) - (1-frequency) * math.log(1-frequency)
#             else:
#                 entropy = 0
            
#             # Combined score
#             score = entropy * filtering_power * complexity_factor
            
#             scored_patterns.append((score, pattern_str, pattern))
        
#         # Sort by score (higher is better)
#         scored_patterns.sort(reverse=True, key=lambda x: x[0])
        
#         # Select top-k
#         selected_patterns = []
#         selected_strings = []
        
#         for score, pattern_str, pattern in scored_patterns[:self.k]:
#             selected_patterns.append(pattern)
#             selected_strings.append(pattern_str)
        
#         # If we don't have enough patterns, add simple ones
#         if len(selected_patterns) < self.k:
#             print(f"Warning: Only found {len(selected_patterns)} discriminative patterns")
#             # Add single-node patterns
#             for label in range(1, 10):  # Assuming labels 1-9 exist
#                 if len(selected_patterns) >= self.k:
#                     break
#                 simple_pattern = {'nodes': {0: label}, 'edges': []}
#                 selected_patterns.append(simple_pattern)
        
#         return selected_patterns[:self.k], selected_strings[:self.k]
    
#     def _parse_pattern_string(self, pattern_str):
#         """Parse canonical pattern string back to dict"""
#         # Format: "N0:1,1:2|E0-1:3"
#         if '|' not in pattern_str:
#             return {'nodes': {0: 1}, 'edges': []}
        
#         nodes_part, edges_part = pattern_str.split('|')
        
#         # Parse nodes
#         nodes = {}
#         if nodes_part.startswith('N'):
#             nodes_str = nodes_part[1:]
#             if nodes_str:
#                 for pair in nodes_str.split(','):
#                     node_id_str, label_str = pair.split(':')
#                     nodes[int(node_id_str)] = int(label_str)
        
#         # Parse edges
#         edges = []
#         if edges_part.startswith('E'):
#             edges_str = edges_part[1:]
#             if edges_str:
#                 for edge_str in edges_str.split(','):
#                     if '-' in edge_str and ':' in edge_str:
#                         nodes_part, label_str = edge_str.split(':')
#                         u_str, v_str = nodes_part.split('-')
#                         edges.append((int(u_str), int(v_str), int(label_str)))
        
#         return {'nodes': nodes, 'edges': edges}


# class FeatureExtractor:
#     """Extract binary features for graphs"""
    
#     def __init__(self, patterns):
#         self.patterns = patterns
#         self.enumerator = DFSEnumerator(max_nodes=4, min_support=0.05)
    
#     def extract_features(self, graphs):
#         """Extract binary feature vectors for graphs"""
#         n_graphs = len(graphs)
#         n_patterns = len(self.patterns)
        
#         features = np.zeros((n_graphs, n_patterns), dtype=np.int8)
        
#         # Process graphs in parallel
#         with Pool(cpu_count()) as pool:
#             results = pool.starmap(
#                 self._process_single_graph,
#                 [(i, graph) for i, graph in enumerate(graphs)]
#             )
        
#         # Combine results
#         for graph_idx, pattern_indices in results:
#             for pattern_idx in pattern_indices:
#                 features[graph_idx, pattern_idx] = 1
        
#         return features
    
#     def _process_single_graph(self, graph_idx, graph):
#         """Process single graph (for parallel execution)"""
#         # Enumerate subgraphs from this graph
#         graph_subgraphs = self.enumerator.enumerate_subgraphs(graph)
        
#         # Check which patterns are present
#         present_patterns = []
#         for pattern_idx, pattern in enumerate(self.patterns):
#             pattern_str = self.enumerator._canonical_label(pattern)
#             if pattern_str in graph_subgraphs:
#                 present_patterns.append(pattern_idx)
        
#         return graph_idx, present_patterns


# def identify_patterns(database_file, patterns_file, fsm_path=None):
#     """Identify discriminative subgraphs without class labels"""
#     print("Step 1: Reading database graphs...")
#     reader = GraphReader()
#     database = reader.read(database_file)
    
#     print(f"Read {len(database)} graphs from database")
    
#     print("Step 2: Removing duplicate graphs...")
#     database = reader.remove_duplicates(database)
#     print(f"After duplicate removal: {len(database)} unique graphs")
    
#     print("Step 3: Enumerating subgraphs (DFS-based)...")
#     enumerator = DFSEnumerator(max_nodes=4, min_support=0.05)
    
#     # Count pattern frequencies across database
#     pattern_counter = Counter()
    
#     # Process graphs in parallel for speed
#     with Pool(cpu_count()) as pool:
#         results = pool.map(enumerator.enumerate_subgraphs, database)
    
#     # Aggregate results
#     for subgraphs in results:
#         pattern_counter.update(subgraphs)
    
#     print(f"Found {len(pattern_counter)} unique subgraph patterns")
    
#     print("Step 4: Selecting top-50 discriminative patterns...")
#     selector = PatternSelector(k=50)
#     selected_patterns, selected_strings = selector.select_patterns(
#         pattern_counter, len(database)
#     )
    
#     print(f"Selected {len(selected_patterns)} patterns")
    
#     print("Step 5: Saving patterns to file...")
#     with open(patterns_file, 'w') as f:
#         for pattern_str in selected_strings:
#             f.write(f"{pattern_str}\n")
    
#     print(f"Patterns saved to {patterns_file}")
#     return selected_patterns


# def convert_to_features(graphs_file, patterns_file, features_file):
#     """Convert graphs to feature vectors"""
#     print("Step 1: Reading graphs...")
#     reader = GraphReader()
#     graphs = reader.read(graphs_file)
#     print(f"Read {len(graphs)} graphs")
    
#     print("Step 2: Loading patterns...")
#     patterns = []
#     pattern_strings = []
#     with open(patterns_file, 'r') as f:
#         for line in f:
#             pattern_str = line.strip()
#             if pattern_str:
#                 pattern_strings.append(pattern_str)
    
#     # Parse pattern strings
#     selector = PatternSelector()
#     for pattern_str in pattern_strings:
#         pattern = selector._parse_pattern_string(pattern_str)
#         patterns.append(pattern)
    
#     print(f"Loaded {len(patterns)} patterns")
    
#     print("Step 3: Extracting features...")
#     extractor = FeatureExtractor(patterns)
#     features = extractor.extract_features(graphs)
    
#     print(f"Feature matrix shape: {features.shape}")
    
#     print("Step 4: Saving features...")
#     np.save(features_file, features)
#     print(f"Features saved to {features_file}")
    
#     return features


# # def generate_candidates(db_features_file, query_features_file, output_file):
# #     """Generate candidate sets"""
# #     print("Loading feature matrices...")
# #     db_features = np.load(db_features_file)
# #     query_features = np.load(query_features_file)
    
# #     print(f"Database features shape: {db_features.shape}")
# #     print(f"Query features shape: {query_features.shape}")
    
# #     n_queries = query_features.shape[0]
# #     n_db = db_features.shape[0]
    
# #     print("Generating candidates...")
# #     with open(output_file, 'w') as f:
# #         for q_idx in range(n_queries):
# #             f.write(f"q # {q_idx + 1}\n")
            
# #             q_vec = query_features[q_idx]
# #             required_features = np.where(q_vec == 1)[0]
            
# #             if len(required_features) == 0:
# #                 # Query has no features → all graphs are candidates
# #                 candidates = list(range(1, n_db + 1))
# #             else:
# #                 # Start with graphs having first required feature
# #                 first_feature = required_features[0]
# #                 candidates = set(np.where(db_features[:, first_feature] == 1)[0])
                
# #                 # Intersect with other required features
# #                 for feature_idx in required_features[1:]:
# #                     candidates.intersection_update(
# #                         set(np.where(db_features[:, feature_idx] == 1)[0])
# #                     )
# #                     if not candidates:
# #                         break
                
# #                 # Convert to 1-based indexing
# #                 candidates = sorted([i + 1 for i in candidates])
            
# #             if candidates:
# #                 f.write(f"c # {' '.join(map(str, candidates))}\n")
# #             else:
# #                 f.write("c #\n")
    
# #     print(f"Candidates saved to {output_file}")
# def generate_candidates(db_features_file, query_features_file, output_file,
#                         min_overlap_ratio=0.3):
#     """
#     FN-safe candidate generation using soft filtering
#     """

#     print("Loading feature matrices...")
#     db_features = np.load(db_features_file)
#     query_features = np.load(query_features_file)

#     n_queries = query_features.shape[0]
#     n_db = db_features.shape[0]

#     print("Generating candidates (FN-safe)...")

#     with open(output_file, 'w') as f:
#         for q_idx in range(n_queries):
#             f.write(f"q # {q_idx + 1}\n")

#             q_vec = query_features[q_idx]
#             q_feats = np.where(q_vec == 1)[0]

#             # No features → everything is candidate
#             if len(q_feats) == 0:
#                 candidates = list(range(1, n_db + 1))
#             else:
#                 min_required = max(1, int(len(q_feats) * min_overlap_ratio))

#                 candidates = []
#                 for g_idx in range(n_db):
#                     overlap = np.sum(db_features[g_idx, q_feats])
#                     if overlap >= min_required:
#                         candidates.append(g_idx + 1)

#             if candidates:
#                 f.write(f"c # {' '.join(map(str, candidates))}\n")
#             else:
#                 f.write("c #\n")

#     print(f"Candidates saved to {output_file}")


# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         if sys.argv[1] == "identify" and len(sys.argv) == 4:
#             identify_patterns(sys.argv[2], sys.argv[3])
#         elif sys.argv[1] == "convert" and len(sys.argv) == 5:
#             convert_to_features(sys.argv[2], sys.argv[3], sys.argv[4])
#         elif sys.argv[1] == "generate" and len(sys.argv) == 5:
#             generate_candidates(sys.argv[2], sys.argv[3], sys.argv[4])




import sys
import os
import numpy as np
from collections import defaultdict, Counter
import itertools
import math
from multiprocessing import Pool, cpu_count
import hashlib
import argparse  # Add this for better argument parsing

#### LLM-assisted code (ChatGPT); all logic and correctness verified by the authors.
# The DFS-based subgraph enumeration and canonical labeling is adapted from Method 4
# but modified for graph indexing without class labels.
####


class GraphReader:
    """Read graphs from dataset files"""
    
    def read(self, filename):
        """Read graphs in the given format"""
        graphs = []
        current = {'nodes': {}, 'edges': []}
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current['nodes'] or current['edges']:
                        graphs.append(current)
                        current = {'nodes': {}, 'edges': []}
                    continue
                
                if line.startswith('#'):
                    if current['nodes'] or current['edges']:
                        graphs.append(current)
                    current = {'nodes': {}, 'edges': []}
                elif line.startswith('v'):
                    parts = line.split()
                    if len(parts) == 3:
                        nid = int(parts[1])
                        label = int(parts[2])
                        current['nodes'][nid] = label
                elif line.startswith('e'):
                    parts = line.split()
                    if len(parts) == 4:
                        u = int(parts[1])
                        v = int(parts[2])
                        label = int(parts[3])
                        current['edges'].append((u, v, label))
        
        if current['nodes'] or current['edges']:
            graphs.append(current)
        
        return graphs
    
    def remove_duplicates(self, graphs):
        """Remove duplicate graphs while preserving order"""
        seen = set()
        unique_graphs = []
        
        for graph in graphs:
            # Create canonical representation
            graph_key = self._graph_to_key(graph)
            if graph_key not in seen:
                seen.add(graph_key)
                unique_graphs.append(graph)
        
        return unique_graphs
    
    def _graph_to_key(self, graph):
        """Create unique key for graph"""
        # Sort nodes by label
        nodes = sorted((nid, label) for nid, label in graph['nodes'].items())
        
        # Sort edges with normalized node order
        edges = []
        for u, v, label in graph['edges']:
            u_label = graph['nodes'][u]
            v_label = graph['nodes'][v]
            if u_label <= v_label:
                edges.append((u_label, v_label, label))
            else:
                edges.append((v_label, u_label, label))
        edges.sort()
        
        return f"N{nodes}E{edges}"


class DFSEnumerator:
    """DFS-based subgraph enumeration with canonical labeling"""
    
    def __init__(self, max_nodes=4, min_support=0.05):
        self.max_nodes = max_nodes
        self.min_support = min_support
    
    def enumerate_subgraphs(self, graph):
        """Enumerate connected subgraphs from a single graph"""
        subgraphs = set()
        
        # Convert to adjacency list for faster neighbor access
        adj_list = self._build_adjacency_list(graph)
        nodes = list(graph['nodes'].keys())
        
        # Start DFS from each node
        for start_node in nodes:
            self._dfs(start_node, [start_node], [], adj_list, graph, subgraphs)
        
        return subgraphs
    
    def _build_adjacency_list(self, graph):
        """Build adjacency list from graph"""
        adj = defaultdict(list)
        for u, v, label in graph['edges']:
            adj[u].append((v, label))
            adj[v].append((u, label))
        return adj
    
    def _dfs(self, current_node, visited_nodes, visited_edges, adj_list, graph, subgraphs):
        """Recursive DFS to grow subgraphs"""
        # Save current subgraph if non-empty
        if len(visited_nodes) >= 1:
            subgraph = self._extract_subgraph(visited_nodes, visited_edges, graph)
            canonical = self._canonical_label(subgraph)
            subgraphs.add(canonical)
        
        # Stop if reached max size
        if len(visited_nodes) >= self.max_nodes:
            return
        
        # Explore neighbors
        for neighbor, edge_label in adj_list[current_node]:
            # Check if we can add this neighbor
            if neighbor in visited_nodes:
                # Already visited, but could add edge if not already present
                edge = (min(current_node, neighbor), max(current_node, neighbor), edge_label)
                if edge not in visited_edges:
                    new_edges = visited_edges + [edge]
                    self._dfs(current_node, visited_nodes, new_edges, adj_list, graph, subgraphs)
            else:
                # New node
                new_nodes = visited_nodes + [neighbor]
                edge = (min(current_node, neighbor), max(current_node, neighbor), edge_label)
                new_edges = visited_edges + [edge]
                self._dfs(neighbor, new_nodes, new_edges, adj_list, graph, subgraphs)
    
    def _extract_subgraph(self, nodes, edges, graph):
        """Extract subgraph from node and edge lists"""
        subgraph_nodes = {}
        node_mapping = {}
        
        # Map original node IDs to sequential IDs
        for i, node_id in enumerate(sorted(nodes)):
            node_mapping[node_id] = i
            subgraph_nodes[i] = graph['nodes'][node_id]
        
        subgraph_edges = []
        for u, v, label in edges:
            new_u = node_mapping[min(u, v)]
            new_v = node_mapping[max(u, v)]
            subgraph_edges.append((new_u, new_v, label))
        
        return {'nodes': subgraph_nodes, 'edges': subgraph_edges}
    
    def _canonical_label(self, subgraph):
        """Create canonical string representation of subgraph"""
        # Sort nodes by label, then by degree
        node_degrees = Counter()
        for u, v, _ in subgraph['edges']:
            node_degrees[u] += 1
            node_degrees[v] += 1
        
        # Create node tuples (label, degree, original_id) for sorting
        node_tuples = []
        for node_id, label in subgraph['nodes'].items():
            degree = node_degrees[node_id]
            node_tuples.append((label, degree, node_id))
        
        # Sort nodes
        node_tuples.sort()
        
        # Create mapping to new IDs
        node_mapping = {}
        new_nodes = {}
        for i, (label, degree, old_id) in enumerate(node_tuples):
            node_mapping[old_id] = i
            new_nodes[i] = label
        
        # Apply mapping to edges and sort
        new_edges = []
        for u, v, label in subgraph['edges']:
            new_u = node_mapping[u]
            new_v = node_mapping[v]
            if new_u <= new_v:
                new_edges.append((new_u, new_v, label))
            else:
                new_edges.append((new_v, new_u, label))
        new_edges.sort()
        
        # Create canonical string
        nodes_str = ','.join(f"{i}:{l}" for i, l in sorted(new_nodes.items()))
        edges_str = ','.join(f"{u}-{v}:{l}" for u, v, l in new_edges)
        
        return f"N{nodes_str}|E{edges_str}"


class PatternSelector:
    """Select discriminative patterns for graph indexing"""
    
    def __init__(self, k=50):
        self.k = k
    
    def select_patterns(self, pattern_frequencies, database_size, k=None):
        """
        Select top-k patterns without class labels
        Uses filtering power instead of class discrimination
        
        Parameters:
        - pattern_frequencies: dict of pattern -> frequency
        - database_size: total number of graphs
        - k: number of patterns to select (default: self.k)
        """
        if k is None:
            k = self.k
        
        scored_patterns = []
        
        for pattern_str, freq in pattern_frequencies.items():
            frequency = freq / database_size
            
            # Skip patterns that are too rare or too common
            # Ideal for filtering: appears in ~50% of graphs
            if frequency < 0.05 or frequency > 0.95:
                continue
            
            # Calculate filtering power
            # Pattern that appears in 50% of graphs can filter 50%
            filtering_power = 1.0 - abs(0.5 - frequency) * 2
            
            # Calculate pattern complexity
            pattern = self._parse_pattern_string(pattern_str)
            complexity = len(pattern['nodes']) + len(pattern['edges'])
            complexity_factor = min(complexity, 5) / 5.0
            
            # Calculate entropy (information content)
            if frequency > 0 and frequency < 1:
                entropy = -frequency * math.log(frequency) - (1-frequency) * math.log(1-frequency)
            else:
                entropy = 0
            
            # Combined score
            score = entropy * filtering_power * complexity_factor
            
            scored_patterns.append((score, pattern_str, pattern, frequency))
        
        # Sort by score (higher is better)
        scored_patterns.sort(reverse=True, key=lambda x: x[0])
        
        # Select top-k
        selected_patterns = []
        selected_strings = []
        selected_frequencies = []
        
        for score, pattern_str, pattern, freq in scored_patterns[:k]:
            selected_patterns.append(pattern)
            selected_strings.append(pattern_str)
            selected_frequencies.append(freq)
        
        # If we don't have enough patterns, add simple ones
        if len(selected_patterns) < k:
            print(f"Warning: Only found {len(selected_patterns)} discriminative patterns, adding simple ones")
            # Add single-node patterns
            for label in range(1, 10):  # Assuming labels 1-9 exist
                if len(selected_patterns) >= k:
                    break
                simple_pattern = {'nodes': {0: label}, 'edges': []}
                simple_str = f"N0:{label}|E"
                selected_patterns.append(simple_pattern)
                selected_strings.append(simple_str)
                selected_frequencies.append(0.5)  # Approximate frequency
        
        return selected_patterns[:k], selected_strings[:k], selected_frequencies[:k]
    
    def _parse_pattern_string(self, pattern_str):
        """Parse canonical pattern string back to dict"""
        # Format: "N0:1,1:2|E0-1:3"
        if '|' not in pattern_str:
            return {'nodes': {0: 1}, 'edges': []}
        
        nodes_part, edges_part = pattern_str.split('|')
        
        # Parse nodes
        nodes = {}
        if nodes_part.startswith('N'):
            nodes_str = nodes_part[1:]
            if nodes_str:
                for pair in nodes_str.split(','):
                    node_id_str, label_str = pair.split(':')
                    nodes[int(node_id_str)] = int(label_str)
        
        # Parse edges
        edges = []
        if edges_part.startswith('E'):
            edges_str = edges_part[1:]
            if edges_str:
                for edge_str in edges_str.split(','):
                    if '-' in edge_str and ':' in edge_str:
                        nodes_part, label_str = edge_str.split(':')
                        u_str, v_str = nodes_part.split('-')
                        edges.append((int(u_str), int(v_str), int(label_str)))
        
        return {'nodes': nodes, 'edges': edges}
    
    def analyze_pattern_distribution(self, pattern_frequencies, database_size):
        """Analyze pattern distribution to suggest optimal k"""
        frequencies = [freq/database_size for freq in pattern_frequencies.values()]
        
        if not frequencies:
            return 50  # Default
        
        # Analyze distribution
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        
        # Patterns with moderate frequency (0.1 to 0.9) are good
        moderate_patterns = sum(0.1 <= f <= 0.9 for f in frequencies)
        
        # Suggested k based on available moderate patterns
        suggested_k = min(100, moderate_patterns)  # Up to 100
        
        print(f"Pattern analysis:")
        print(f"  Total patterns: {len(frequencies)}")
        print(f"  Mean frequency: {mean_freq:.3f}")
        print(f"  Std frequency: {std_freq:.3f}")
        print(f"  Moderate patterns (0.1-0.9): {moderate_patterns}")
        print(f"  Suggested k: {suggested_k}")
        
        return max(10, min(suggested_k, 100))  # Between 10 and 100


class FeatureExtractor:
    """Extract binary features for graphs"""
    
    def __init__(self, patterns):
        self.patterns = patterns
        self.enumerator = DFSEnumerator(max_nodes=4, min_support=0.05)
    
    def extract_features(self, graphs):
        """Extract binary feature vectors for graphs"""
        n_graphs = len(graphs)
        n_patterns = len(self.patterns)
        
        features = np.zeros((n_graphs, n_patterns), dtype=np.int8)
        
        # Process graphs in parallel
        with Pool(cpu_count()) as pool:
            results = pool.starmap(
                self._process_single_graph,
                [(i, graph) for i, graph in enumerate(graphs)]
            )
        
        # Combine results
        for graph_idx, pattern_indices in results:
            for pattern_idx in pattern_indices:
                features[graph_idx, pattern_idx] = 1
        
        return features
    
    def _process_single_graph(self, graph_idx, graph):
        """Process single graph (for parallel execution)"""
        # Enumerate subgraphs from this graph
        graph_subgraphs = self.enumerator.enumerate_subgraphs(graph)
        
        # Check which patterns are present
        present_patterns = []
        for pattern_idx, pattern in enumerate(self.patterns):
            pattern_str = self.enumerator._canonical_label(pattern)
            if pattern_str in graph_subgraphs:
                present_patterns.append(pattern_idx)
        
        return graph_idx, present_patterns


def identify_patterns(database_file, patterns_file, k=50, fsm_path=None):
    """Identify discriminative subgraphs without class labels
    
    Parameters:
    - database_file: path to database graphs
    - patterns_file: path to output patterns
    - k: number of patterns to select (default: 50)
    - fsm_path: optional path to FSM executable
    """
    print(f"Identifying top-{k} discriminative patterns...")
    
    print("Step 1: Reading database graphs...")
    reader = GraphReader()
    database = reader.read(database_file)
    
    print(f"Read {len(database)} graphs from database")
    
    print("Step 2: Removing duplicate graphs...")
    database = reader.remove_duplicates(database)
    print(f"After duplicate removal: {len(database)} unique graphs")
    
    print("Step 3: Enumerating subgraphs (DFS-based)...")
    enumerator = DFSEnumerator(max_nodes=4, min_support=0.05)
    
    # Count pattern frequencies across database
    pattern_counter = Counter()
    
    # Process graphs in parallel for speed
    with Pool(cpu_count()) as pool:
        results = pool.map(enumerator.enumerate_subgraphs, database)
    
    # Aggregate results
    for subgraphs in results:
        pattern_counter.update(subgraphs)
    
    print(f"Found {len(pattern_counter)} unique subgraph patterns")
    
    print("Step 4: Selecting top-k discriminative patterns...")
    selector = PatternSelector(k=k)
    
    # Optional: Analyze pattern distribution
    suggested_k = selector.analyze_pattern_distribution(pattern_counter, len(database))
    if suggested_k != k:
        print(f"Note: Based on pattern distribution, k={suggested_k} might be better")
    
    selected_patterns, selected_strings, frequencies = selector.select_patterns(
        pattern_counter, len(database), k=k
    )
    
    print(f"Selected {len(selected_patterns)} patterns")
    
    # Print statistics
    if frequencies:
        avg_freq = np.mean(frequencies)
        print(f"Average pattern frequency: {avg_freq:.3f}")
        print(f"Pattern frequency range: {min(frequencies):.3f} - {max(frequencies):.3f}")
    
    print("Step 5: Saving patterns to file...")
    with open(patterns_file, 'w') as f:
        # First line: k value (for documentation)
        f.write(f"# k={k}\n")
        for pattern_str in selected_strings:
            f.write(f"{pattern_str}\n")
    
    print(f"Patterns saved to {patterns_file}")
    
    # Save pattern statistics
    stats_file = patterns_file + ".stats"
    with open(stats_file, 'w') as f:
        f.write(f"k={k}\n")
        f.write(f"total_patterns_found={len(pattern_counter)}\n")
        f.write(f"selected_patterns={len(selected_patterns)}\n")
        if frequencies:
            f.write(f"avg_frequency={np.mean(frequencies):.4f}\n")
            f.write(f"min_frequency={min(frequencies):.4f}\n")
            f.write(f"max_frequency={max(frequencies):.4f}\n")
    
    return selected_patterns


def convert_to_features(graphs_file, patterns_file, features_file):
    """Convert graphs to feature vectors"""
    print("Step 1: Reading graphs...")
    reader = GraphReader()
    graphs = reader.read(graphs_file)
    print(f"Read {len(graphs)} graphs")
    
    print("Step 2: Loading patterns...")
    patterns = []
    pattern_strings = []
    with open(patterns_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip comments
                pattern_strings.append(line)
    
    # Parse pattern strings
    selector = PatternSelector()
    for pattern_str in pattern_strings:
        pattern = selector._parse_pattern_string(pattern_str)
        patterns.append(pattern)
    
    print(f"Loaded {len(patterns)} patterns")
    
    print("Step 3: Extracting features...")
    extractor = FeatureExtractor(patterns)
    features = extractor.extract_features(graphs)
    
    print(f"Feature matrix shape: {features.shape}")
    
    print("Step 4: Saving features...")
    np.save(features_file, features)
    print(f"Features saved to {features_file}")
    
    return features


def generate_candidates(db_features_file, query_features_file, output_file,
                        min_overlap_ratio=0.3, verbose=False):
    """
    FN-safe candidate generation using soft filtering
    
    Parameters:
    - min_overlap_ratio: minimum ratio of query features that must be present
    - verbose: print detailed statistics
    """
    print("Loading feature matrices...")
    db_features = np.load(db_features_file)
    query_features = np.load(query_features_file)

    n_queries = query_features.shape[0]
    n_db = db_features.shape[0]
    k = db_features.shape[1]  # Number of features/patterns

    print(f"Database: {n_db} graphs, {k} features")
    print(f"Queries: {n_queries} graphs")
    print(f"Using min_overlap_ratio = {min_overlap_ratio}")

    print("Generating candidates (FN-safe)...")

    candidate_counts = []
    with open(output_file, 'w') as f:
        for q_idx in range(n_queries):
            f.write(f"q # {q_idx + 1}\n")

            q_vec = query_features[q_idx]
            q_feats = np.where(q_vec == 1)[0]

            # No features → everything is candidate
            if len(q_feats) == 0:
                candidates = list(range(1, n_db + 1))
            else:
                # Dynamic min_required: at least 1, or ratio of query features
                min_required = max(1, int(len(q_feats) * min_overlap_ratio))
                
                if verbose:
                    print(f"Query {q_idx+1}: {len(q_feats)} features, min_required={min_required}")

                candidates = []
                for g_idx in range(n_db):
                    overlap = np.sum(db_features[g_idx, q_feats])
                    if overlap >= min_required:
                        candidates.append(g_idx + 1)
            
            candidate_counts.append(len(candidates))
            
            if candidates:
                f.write(f"c # {' '.join(map(str, candidates))}\n")
            else:
                f.write("c #\n")
    
    # Print statistics
    if candidate_counts:
        avg_candidates = np.mean(candidate_counts)
        max_candidates = max(candidate_counts)
        min_candidates = min(candidate_counts)
        
        print(f"Candidate set statistics:")
        print(f"  Average size: {avg_candidates:.1f}")
        print(f"  Minimum size: {min_candidates}")
        print(f"  Maximum size: {max_candidates}")
        print(f"  Filtering ratio: {avg_candidates/n_db:.3%}")
    
    print(f"Candidates saved to {output_file}")


def evaluate_candidate_sets(candidates_file, true_matches_file):
    """
    Evaluate candidate sets (for testing only)
    Requires knowing true matches R_q
    """
    # This is for testing/evaluation purposes
    # In real competition, you won't have true_matches_file
    pass


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Graph Indexing System")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # identify command
    parser_identify = subparsers.add_parser('identify', help='Identify discriminative patterns')
    parser_identify.add_argument('database_file', help='Path to database graphs')
    parser_identify.add_argument('patterns_file', help='Path to output patterns')
    parser_identify.add_argument('--k', type=int, default=50, help='Number of patterns to select')
    parser_identify.add_argument('--max-nodes', type=int, default=4, help='Maximum subgraph size')
    parser_identify.add_argument('--min-support', type=float, default=0.05, help='Minimum support')
    
    # convert command
    parser_convert = subparsers.add_parser('convert', help='Convert graphs to features')
    parser_convert.add_argument('graphs_file', help='Path to graphs')
    parser_convert.add_argument('patterns_file', help='Path to patterns')
    parser_convert.add_argument('features_file', help='Path to output features')
    
    # generate command
    parser_generate = subparsers.add_parser('generate', help='Generate candidate sets')
    parser_generate.add_argument('db_features', help='Path to database features')
    parser_generate.add_argument('query_features', help='Path to query features')
    parser_generate.add_argument('output_file', help='Path to output candidates')
    parser_generate.add_argument('--min-overlap', type=float, default=0.3, 
                                help='Minimum overlap ratio (0.0-1.0)')
    parser_generate.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.command == 'identify':
        # Update enumerator parameters if provided
        global DFSEnumerator
        DFSEnumerator = type('DFSEnumerator', DFSEnumerator.__bases__, dict(DFSEnumerator.__dict__))
        DFSEnumerator.__init__ = lambda self, max_nodes=args.max_nodes, min_support=args.min_support: \
            DFSEnumerator.__init__(self, max_nodes, min_support)
        
        identify_patterns(args.database_file, args.patterns_file, k=args.k)
    
    elif args.command == 'convert':
        convert_to_features(args.graphs_file, args.patterns_file, args.features_file)
    
    elif args.command == 'generate':
        generate_candidates(args.db_features, args.query_features, args.output_file,
                           min_overlap_ratio=args.min_overlap, verbose=args.verbose)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    # For backward compatibility with shell scripts
    if len(sys.argv) > 1 and sys.argv[1] in ["identify", "convert", "generate"]:
        main()
    else:
        # Old interface for shell scripts
        if len(sys.argv) > 1:
            if sys.argv[1] == "identify" and len(sys.argv) == 4:
                identify_patterns(sys.argv[2], sys.argv[3])
            elif sys.argv[1] == "identify" and len(sys.argv) == 5:
                # With k parameter
                k = int(sys.argv[4]) if sys.argv[4].isdigit() else 50
                identify_patterns(sys.argv[2], sys.argv[3], k=k)
            elif sys.argv[1] == "convert" and len(sys.argv) == 5:
                convert_to_features(sys.argv[2], sys.argv[3], sys.argv[4])
            elif sys.argv[1] == "generate" and len(sys.argv) == 5:
                generate_candidates(sys.argv[2], sys.argv[3], sys.argv[4])
            elif sys.argv[1] == "generate" and len(sys.argv) == 6:
                # With min_overlap parameter
                min_overlap = float(sys.argv[5])
                generate_candidates(sys.argv[2], sys.argv[3], sys.argv[4], 
                                   min_overlap_ratio=min_overlap)