import torch
import torch.nn as nn
import os

class Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        tmp = (weight == weight.max(dim=0, keepdim=True)[0])
        sigma = tmp.view_as(weight).int().float()
        sigma_w = sigma.mul(weight)
        #output = input.mm(sigma.mul(weight).t())
        output = input.mm(sigma_w)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        sigma = (weight == weight.max(dim=0, keepdim=True)[0]).view_as(input).int().float()
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(sigma.mul(weight))
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.mm(input.mm(sigma))
        return grad_input, grad_weight, None

class ClassProjection(nn.Module):
    def __init__(self, mapfile_path = None, treeNode = None, learnable = False, n_super = 5, n_sub = 10,
                intermap = None):
        super(ClassProjection, self).__init__()
        tmp_pair = {} # {subclass, motherclass}
        i_mother_list = []
        i_child_list = []
        self.input_dim = 0 
        self.output_dim = 0
        
        if mapfile_path is not None:
            with (mapfile_path, 'r') as f:
                for ln in f:
                    fields = [int(field) for field in ln.rstrip('\n').split(',')]
                    tmp_pair[fields[1]] = fields[0]
                    if fields[0] not in i_mother_list:
                        self.input_dim += 1
                        i_mother_list.append(fields[0])
                    self.output_dim += 1
        elif treeNode is not None:
            for node in treeNode:
                tmp_pair[node[1]] = node[0]
                if node[0] not in i_mother_list:
                    self.input_dim += 1
                    i_mother_list.append(node[0])
                self.output_dim += 1
        else:
            self.input_dim = n_super
            self.output_dim = n_sub
            self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
            self.weight.data.uniform_(0.0, 1.0)
            if not learnable:
                self.weight.requires_grad = False
            print('The weight will be auto initialized.')
        if mapfile_path is not None or treeNode is not None:
            self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
            self.weight.data.zero_()
            if not learnable:
                self.weight.requires_grad = False
            if intermap is None:
                for subcls in tmp_pair.keys():
                    self.weight.data[tmp_pair[subcls], subcls] = 1.0
            else:
                for subcls in tmp_pair.keys():
                    for basecls in intermap[subcls]:
                        self.weight.data[tmp_pair[subcls], basecls] = 1.0
    
    def forward(self, input):
        return Projection.apply(input, self.weight)
    
    #def extra_repr(self):
        #return None
