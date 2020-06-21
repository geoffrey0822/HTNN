# +
import numpy as np
import os

def build_itree(tree_path):
    output = {}
    n_coarst = 0
    coarst_output_lens = []
    tmp_dict = []
    with open(tree_path, 'r') as f:
        for ln in f:
            nodes = [int(node) for node in ln.rstrip('\n').split(',')]
            if nodes[-1] not in output:
                output[nodes[-1]] = []
            n_node = len(nodes)
            if n_coarst == 0:
                n_coarst = n_node-1
                for j in range(n_node-1):
                    tmp_dict.append([])
            for i in range(n_node-1):
                output[nodes[-1]].append(nodes[n_node-i-2])
                if nodes[n_node-i-2] not in tmp_dict[i]:
                    tmp_dict[i].append(nodes[n_node-i-2])
    for i_coarst in range(n_coarst):
        coarst_output_lens.append(len(tmp_dict[i_coarst]))
    return output, n_coarst, coarst_output_lens
