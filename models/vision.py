import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from . layers import ClassProjection


class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh()
        )
        
        self.input_dim = [3, 32, 32]
        self.feat_dim = 84


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        #logits = self.classifier(x)
        #probs = F.softmax(logits, dim=1)
        return self.fc(x)


class HTCNN(nn.Module):
    def __init__(self, classTree_path, with_aux = True, with_fc = True, backbone = None, 
                 feat_dim = 0, isCuda = False):
        super(HTCNN, self).__init__()
        
        
        self.n_bin = 0
        bins = []
        bin_uniques = []
        i_bins = []
        self.with_aux = with_aux
        self.with_fc = with_fc
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
        
        output_dim = len(bin_uniques[-1])
        if backbone is not None:
            self.backbone_nn = backbone
            input_dim = self.backbone_nn.feat_dim
        else:
            self.backbone_nn = None
            input_dim = feat_dim
            if input_dim == 0:
                input_dim = 128
                
            self.fc = True
        
        self.proj_layers = []
        self.fc_s = []
        i = 0
        for ibin in bins:
            output_dim = len(bin_uniques[i])
            if isCuda:
                self.proj_layers.append(ClassProjection(treeNode = ibin, intermap=i_bins[i]).cuda())
                if with_fc:
                    self.fc_s.append(nn.Linear(input_dim, output_dim).cuda())
            else:
                self.proj_layers.append(ClassProjection(treeNode = ibin, intermap=i_bins[i]))
                if with_fc:
                    self.fc_s.append(nn.Linear(input_dim, output_dim))
            
            i += 1
            
        #define back-bone network layers
        if with_fc:
            output_dim = len(bin_uniques[-1])
            if isCuda:
                self.fc_s.append(nn.Linear(input_dim, output_dim).cuda())
            else:
                self.fc_s.append(nn.Linear(input_dim, output_dim))
    
    def backbone(self, x):
        if self.backbone_nn is not None:
            return self.backbone_nn(x)
        else:
            return x
    
    def forward(self, x):
        # output nodes should be in ordered {coarst 1, coarst 2, ..., coarst n, fine} for n-coarst problem
        if self.with_fc:
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
        else:
            if self.with_aux:
                y = []
                y_ = None
                for i in range(self.n_bin):
                    _y = F.softmax(self.backbone(x), dim=1)
                    y.append(_y)
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        #y.add(self.proj_layers[i](F.softmax(self.fc_s[i](x)))) # sum
                        y_ = y_.mul(self.proj_layers[i](_y)) # elementwise product
                b_y = F.softmax(self.backbone(x), dim=1)
                y.append(b_y)
                y_ = y_.mul(b_y)
                return y_, y
            else:
                y_ = None
                for i in range(self.n_bin):
                    _y = F.softmax(self.backbone(x), dim=1)
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        #y.add(self.proj_layers[i](F.softmax(self.fc_s[i](x)))) # sum
                        y_ = y_.mul(self.proj_layers[i](_y)) # elementwise product
                b_y = F.softmax(self.backbone(x), dim=1)
                y_ = y_.mul(b_y)
                return y_, None

