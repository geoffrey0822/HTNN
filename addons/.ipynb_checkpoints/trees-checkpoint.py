# +
import numpy as np
import os

def build_itree(tree_path):
    output = {}
    with open(tree_path, 'r') as f:
        for ln in f:
            nodes = [int(node) for node in ln.rstrip('\n').split(',')]
            if nodes[-1] not in output:
                output[nodes[-1]] = []
            n_node = len(nodes)
            for i in range(n_node-1):
                output[nodes[-1]].append(nodes[n_node-i-2])
    return output
