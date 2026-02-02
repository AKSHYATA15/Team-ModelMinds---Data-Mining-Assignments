import numpy as np

def read_graphs_simple(filename):
    """Simple graph reader for quick testing"""
    graphs = []
    current = {'nodes': {}, 'edges': []}
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('#'):
                if current['nodes'] or current['edges']:
                    graphs.append(current)
                    current = {'nodes': {}, 'edges': []}
            elif line.startswith('v'):
                parts = line.split()
                if len(parts) >= 3:
                    nid = int(parts[1])
                    label = int(parts[2])
                    current['nodes'][nid] = label
            elif line.startswith('e'):
                parts = line.split()
                if len(parts) >= 4:
                    u = int(parts[1])
                    v = int(parts[2])
                    label = int(parts[3])
                    current['edges'].append((u, v, label))
    
    if current['nodes'] or current['edges']:
        graphs.append(current)
    
    return graphs


def get_stats(graphs):
    """Get basic statistics about graphs"""
    if not graphs:
        return {}
    
    n_nodes = [len(g['nodes']) for g in graphs]
    n_edges = [len(g['edges']) for g in graphs]
    
    return {
        'count': len(graphs),
        'avg_nodes': np.mean(n_nodes),
        'avg_edges': np.mean(n_edges)
    }