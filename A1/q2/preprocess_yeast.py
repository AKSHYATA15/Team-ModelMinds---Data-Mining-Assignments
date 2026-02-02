#llm generated code 
# preprocess_yeast.py
import os
import sys

LABEL_MAP = {
    'Br': 0, 'C': 1, 'Cl': 2, 'F': 3, 'H': 4,
    'I': 5, 'N': 6, 'O': 7, 'P': 8, 'S': 9, 'Si': 10
}

def parse_yeast(path):
    graphs = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("#"):
                continue

            n = int(f.readline().strip())
            nodes = [LABEL_MAP[f.readline().strip()] for _ in range(n)]

            m = int(f.readline().strip())
            edges = [tuple(map(int, f.readline().split())) for _ in range(m)]

            graphs.append((nodes, edges))
    return graphs

def write_gspan(graphs, out):
    with open(out, "w") as f:
        for i, (nodes, edges) in enumerate(graphs):
            f.write(f"t # {i}\n")
            for j, l in enumerate(nodes):
                f.write(f"v {j} {l}\n")
            for u, v, e in edges:
                f.write(f"e {u} {v} {e}\n")

def write_gaston(graphs, out):
    with open(out, "w") as f:
        for i, (nodes, edges) in enumerate(graphs):
            f.write(f"# {i}\n{len(nodes)}\n")
            for l in nodes:
                f.write(f"{l}\n")
            f.write(f"{len(edges)}\n")
            for u, v, e in edges:
                f.write(f"{u} {v} {e}\n")

if __name__ == "__main__":
    inp, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    g = parse_yeast(inp)
    write_gspan(g, f"{outdir}/yeast_gspan.txt")
    write_gspan(g, f"{outdir}/yeast_fsg.txt")      # FSG uses the same input format as gSpan
    write_gaston(g, f"{outdir}/yeast_gaston.txt")
