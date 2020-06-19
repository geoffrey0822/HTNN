import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from . layers import ClassProjection


class FeatureANN(nn.Module):
    def __init__(self, input_dim, classTree_path, with_aux = True):
        super(FeatureANN, self).__init__()
        self.n_bin = 0
        bins = []
        bin_uniques = []
        i_bins = []
        self.with_aux = with_aux
        with open(classTree_path, 'r') as f:
            for ln in f:
                nodes = [int(field) for field in ln.rstrip('\n').split(',')]
                n_node = len(nodes)
                if bins == []:
                    for i in range(1, n_node):
                        bins.append([])
                        bin_uniques.append([])
                        i_bins.append({})
                        self.n_bin += 1
                    bin_uniques.append([])
                for i in range(1, n_node):
                    bins[i-1].append([nodes[i-1], nodes[i]])
                    if nodes[i-1] not in bin_uniques[i-1]:
                        bin_uniques[i-1].append(nodes[i-1])
                    if nodes[i] not in i_bins[i-1]:
                        i_bins[i-1][nodes[i]] = []
                    i_bins[i-1][nodes[i]].append(nodes[-1])
                if nodes[-1] not in bin_uniques[-1]:
                    bin_uniques[-1].append(nodes[-1])
        
        self.proj_layers = []
        self.fc_s = []
        i = 0
        for ibin in bins:
            output_dim = len(bin_uniques[i])
            self.proj_layers.append(ClassProjection(treeNode = ibin, intermap=i_bins[i]))
            self.fc_s.append(nn.Linear(input_dim, output_dim))
            i += 1
            
        #define back-bone network layers
        output_dim = len(bin_uniques[-1])
        self.fc_s.append(nn.Linear(input_dim, output_dim))
    
    def backbone(self, x):
        return x
    
    def forward(self, x):
        if self.with_aux:
            y = []
            y_ = None
            for i in range(self.n_bin):
                _y = F.softmax(self.fc_s[i](self.backbone(x)), dim=1)
                y.append(_y)
                if y_ is None:
                    y_ = self.proj_layers[i](_y)
                else:
                    #y.add(self.proj_layers[i](F.softmax(self.fc_s[i](x)))) # sum
                    y_ = y_.mul(self.proj_layers[i](_y)) # elementwise product
            b_y = F.softmax(self.fc_s[-1](self.backbone(x)), dim=1)
            y.append(b_y)
            y_ = y_.mul(b_y)
            return y_, y
        else:
            y_ = None
            for i in range(self.n_bin):
                _y = F.softmax(self.fc_s[i](self.backbone(x)), dim=1)
                if y_ is None:
                    y_ = self.proj_layers[i](_y)
                else:
                    #y.add(self.proj_layers[i](F.softmax(self.fc_s[i](x)))) # sum
                    y_ = y_.mul(self.proj_layers[i](_y)) # elementwise product
            b_y = F.softmax(self.fc_s[-1](self.backbone(x)), dim=1)
            y_ = y_.mul(b_y)
            return y_, None
        
