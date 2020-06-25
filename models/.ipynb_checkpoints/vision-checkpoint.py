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
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=256, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=n_classes),
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=128),
            nn.Tanh()
        )
        
        self.input_dim = [3, 32, 32]
        self.feat_dim = 128


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        #logits = self.classifier(x)
        #probs = F.softmax(logits, dim=1)
        return self.fc(x), self.classifier(x)


class AlexNet(nn.Module):

    def __init__(self, n_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        
        self.input_dim = [3, 224, 224]
        self.feat_dim = 4096

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #x = self.classifier(x)
        return self.fc(x), self.classifier(x)


class HTCNN(nn.Module):
    def __init__(self, classTree_path, with_aux = True, with_fc = True, backbone = None, 
                 feat_dim = 0, isCuda = False, isConditionProb = True):
        super(HTCNN, self).__init__()
        
        
        self.n_bin = 0
        bins = []
        bin_uniques = []
        i_bins = []
        self.with_aux = with_aux
        self.with_fc = with_fc
        self.isConditionProb = isConditionProb
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
        
        self.proj_layers = nn.ModuleList()
        self.fc_s = nn.ModuleList()
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
        feat_y, b_y = self.backbone(x)
        if self.with_fc:
            if self.with_aux:
                y = []
                y_ = None
                for i in range(self.n_bin):
                    _y = self.fc_s[i](feat_y)
                    y.append(_y)
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        if self.isConditionProb:
                            y_ = torch.mul(y_, self.proj_layers[i](_y)) # elementwise product
                        else:
                            y_ = torch.add(y_, self.proj_layers[i](_y)) # sum
                y.append(b_y)
                if self.isConditionProb:
                    y_ = torch.mul(y_, b_y)
                else:
                    y_ = torch.add(y_, b_y)
                return y_, y
            else:
                y_ = None
                for i in range(self.n_bin):
                    _y = self.fc_s[i](feat_y)
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        if self.isConditionProb:
                            y_ = torch.mul(y_, self.proj_layers[i](_y)) # elementwise product
                        else:
                            y_ = torch.add(y_, self.proj_layers[i](_y)) # sum

                if self.isConditionProb:
                    y_ = torch.mul(y_, b_y)
                else:
                    y_ = torch.add(y_, b_y)
                return y_, None
        else:
            if self.with_aux:
                y = []
                y_ = None
                for i in range(self.n_bin):
                    _y = b_y
                    y.append(_y)
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        if self.isConditionProb:
                            y_ = torch.mul(y_, self.proj_layers[i](_y)) # elementwise product
                        else:
                            y_ = torch.add(y_, self.proj_layers[i](_y)) # sum
                y.append(b_y)
                if self.isConditionProb:
                    y_ = torch.mul(y_, b_y)
                else:
                    y_ = torch.add(y_, b_y)
                return y_, y
            else:
                y_ = None
                for i in range(self.n_bin):
                    _y = b_y
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        if self.isConditionProb:
                            y_ = torch.mul(y_, self.proj_layers[i](_y)) # elementwise product
                        else:
                            y_ = torch.add(y_, self.proj_layers[i](_y)) # sum
                if self.isConditionProb:
                    y_ = torch.mul(y_, b_y)
                else:
                    y_ = torch.add(y_, b_y)
                return y_, None


class HTCNN_M(nn.Module):
    def __init__(self, classTree_path, with_aux = True, with_fc = True, backbones = None, 
                 feat_dim = [], isCuda = False, isConditionProb = True):
        super(HTCNN_M, self).__init__()
        
        
        self.n_bin = 0
        bins = []
        bin_uniques = []
        i_bins = []
        self.with_aux = with_aux
        self.with_fc = with_fc
        self.isConditionProb = isConditionProb
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
        self.input_dims = []
        if backbones is not None:
            self.backbones = backbones
            for backbone in self.backbones:
                self.input_dims.append(backbone.feat_dim)
        else:
            self.backbones = None
            for input_dim in feat_dim:
                self.input_dims.append(input_dim)
            self.fc = True
        
        self.proj_layers = nn.ModuleList()
        self.fc_s = nn.ModuleList()
        i = 0
        for ibin in bins:
            output_dim = len(bin_uniques[i])
            if isCuda:
                self.proj_layers.append(ClassProjection(treeNode = ibin, intermap=i_bins[i]).cuda())
                if with_fc:
                    self.fc_s.append(nn.Linear(self.input_dims[i], output_dim).cuda())
            else:
                self.proj_layers.append(ClassProjection(treeNode = ibin, intermap=i_bins[i]))
                if with_fc:
                    self.fc_s.append(nn.Linear(self.input_dims[i], output_dim))
            
            i += 1
            
        #define back-bone network layers
        if with_fc:
            output_dim = len(bin_uniques[-1])
            if isCuda:
                self.fc_s.append(nn.Linear(self.input_dims[-1], output_dim).cuda())
            else:
                self.fc_s.append(nn.Linear(self.input_dims[-1], output_dim))
                
        if (len(bins)+1)!=len(self.backbones):
            raise Exception('(%d vs %d)The number of backbone networks must be the same as number of bin'%(len(bins)+1,len(self.backbones)))
    
    def backbone(self, x):
        output = [] 
        if self.backbones is not None:
            for backbone in self.backbones:
                output.append(backbone(x))
            return output
        else:
            return x
    
    def forward(self, x):
        # output nodes should be in ordered {coarst 1, coarst 2, ..., coarst n, fine} for n-coarst problem
        back_results = self.backbone(x)
        if self.with_fc:
            if self.with_aux:
                y = []
                y_ = None
                for i in range(self.n_bin):
                    feat_y, b_y = back_results[i]
                    _y = self.fc_s[i](feat_y)
                    y.append(_y)
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        if self.isConditionProb:
                            y_ = torch.mul(y_, self.proj_layers[i](_y)) # elementwise product
                        else:
                            y_ = torch.add(y_, self.proj_layers[i](_y)) # sum
                b_y = back_results[-1][1]
                y.append(b_y)
                if self.isConditionProb:
                    y_ = torch.mul(y_, b_y)
                else:
                    y_ = torch.add(y_, b_y)
                return y_, y
            else:
                y_ = None
                for i in range(self.n_bin):
                    _y = self.fc_s[i](feat_y)
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        if self.isConditionProb:
                            y_ = torch.mul(y_, self.proj_layers[i](_y)) # elementwise product
                        else:
                            y_ = torch.add(y_, self.proj_layers[i](_y)) # sum
                b_y = back_results[-1][1]
                if self.isConditionProb:
                    y_ = torch.mul(y_, b_y)
                else:
                    y_ = torch.add(y_, b_y)
                return y_, None
        else:
            if self.with_aux:
                y = []
                y_ = None
                for i in range(self.n_bin):
                    feat_y, b_y = back_results[i]
                    _y = b_y
                    y.append(_y)
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        if self.isConditionProb:
                            y_ = torch.mul(y_, self.proj_layers[i](_y)) # elementwise product
                        else:
                            y_ = torch.add(y_, self.proj_layers[i](_y)) # sum
                b_y = back_results[-1][1]
                y.append(b_y)
                if self.isConditionProb:
                    y_ = torch.mul(y_, b_y)
                else:
                    y_ = torch.add(y_, b_y)
                return y_, y
            else:
                y_ = None
                for i in range(self.n_bin):
                    feat_y, b_y = back_results[i]
                    _y = b_y
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        if self.isConditionProb:
                            y_ = torch.mul(y_, self.proj_layers[i](_y)) # elementwise product
                        else:
                            y_ = torch.add(y_, self.proj_layers[i](_y)) # sum
                b_y = back_results[-1][1]
                if self.isConditionProb:
                    y_ = torch.mul(y_, b_y)
                else:
                    y_ = torch.add(y_, b_y)
                return y_, None


class HTCNN_M_IN(nn.Module):
    def __init__(self, classTree_path, with_aux = True, with_fc = True, backbones = None, 
                 feat_dim = [], isCuda = False, isConditionProb = True):
        super(HTCNN_M_IN, self).__init__()
        
        
        self.n_bin = 0
        bins = []
        bin_uniques = []
        i_bins = []
        self.with_aux = with_aux
        self.with_fc = with_fc
        self.isConditionProb = isConditionProb
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
        self.input_dims = []
        if backbones is not None:
            self.backbones = backbones
            for backbone in self.backbones:
                self.input_dims.append(backbone.feat_dim)
        else:
            self.backbones = None
            for input_dim in feat_dim:
                self.input_dims.append(input_dim)
            self.fc = True
        
        self.proj_layers = nn.ModuleList()
        self.fc_s = nn.ModuleList()
        i = 0
        for ibin in bins:
            output_dim = len(bin_uniques[i])
            if isCuda:
                self.proj_layers.append(ClassProjection(treeNode = ibin, intermap=i_bins[i]).cuda())
                if with_fc:
                    self.fc_s.append(nn.Linear(self.input_dims[i], output_dim).cuda())
            else:
                self.proj_layers.append(ClassProjection(treeNode = ibin, intermap=i_bins[i]))
                if with_fc:
                    self.fc_s.append(nn.Linear(self.input_dims[i], output_dim))
            
            i += 1
            
        #define back-bone network layers
        if with_fc:
            output_dim = len(bin_uniques[-1])
            if isCuda:
                self.fc_s.append(nn.Linear(self.input_dims[-1], output_dim).cuda())
            else:
                self.fc_s.append(nn.Linear(self.input_dims[-1], output_dim))
                
        if (len(bins)+1)!=len(self.backbones):
            raise Exception('(%d vs %d)The number of backbone networks must be the same as number of bin'%(len(bins)+1,len(self.backbones)))
    
    def backbone(self, x):
        output = [] 
        if self.backbones is not None:
            i = 0
            for backbone in self.backbones:
                output.append(backbone(x[i]))
                i += 1
            return output
        else:
            return x
    
    def forward(self, x):
        # output nodes should be in ordered {coarst 1, coarst 2, ..., coarst n, fine} for n-coarst problem
        back_results = self.backbone(x)
        if self.with_fc:
            if self.with_aux:
                y = []
                y_ = None
                for i in range(self.n_bin):
                    feat_y, b_y = back_results[i]
                    _y = self.fc_s[i](feat_y)
                    y.append(_y)
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        if self.isConditionProb:
                            y_ = torch.mul(y_, self.proj_layers[i](_y)) # elementwise product
                        else:
                            y_ = torch.add(y_, self.proj_layers[i](_y)) # sum
                b_y = back_results[-1][1]
                y.append(b_y)
                if self.isConditionProb:
                    y_ = torch.mul(y_, b_y)
                else:
                    y_ = torch.add(y_, b_y)
                return y_, y
            else:
                y_ = None
                for i in range(self.n_bin):
                    _y = self.fc_s[i](feat_y)
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        if self.isConditionProb:
                            y_ = torch.mul(y_, self.proj_layers[i](_y)) # elementwise product
                        else:
                            y_ = torch.add(y_, self.proj_layers[i](_y)) # sum
                b_y = back_results[-1][1]
                if self.isConditionProb:
                    y_ = torch.mul(y_, b_y)
                else:
                    y_ = torch.add(y_, b_y)
                return y_, None
        else:
            if self.with_aux:
                y = []
                y_ = None
                for i in range(self.n_bin):
                    feat_y, b_y = back_results[i]
                    _y = b_y
                    y.append(_y)
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        if self.isConditionProb:
                            y_ = torch.mul(y_, self.proj_layers[i](_y)) # elementwise product
                        else:
                            y_ = torch.add(y_, self.proj_layers[i](_y)) # sum
                b_y = back_results[-1][1]
                y.append(b_y)
                if self.isConditionProb:
                    y_ = torch.mul(y_, b_y)
                else:
                    y_ = torch.add(y_, b_y)
                return y_, y
            else:
                y_ = None
                for i in range(self.n_bin):
                    feat_y, b_y = back_results[i]
                    _y = b_y
                    if y_ is None:
                        y_ = self.proj_layers[i](_y)
                    else:
                        if self.isConditionProb:
                            y_ = torch.mul(y_, self.proj_layers[i](_y)) # elementwise product
                        else:
                            y_ = torch.add(y_, self.proj_layers[i](_y)) # sum
                b_y = back_results[-1][1]
                if self.isConditionProb:
                    y_ = torch.mul(y_, b_y)
                else:
                    y_ = torch.add(y_, b_y)
                return y_, None
