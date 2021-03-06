# +
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate,
    GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    RandomCrop, Resize, GridDropout, Compose, Normalize, PadIfNeeded
)
from albumentations.pytorch import ToTensor, ToTensorV2
#from torch.utils.tensorboard import SummaryWriter
import utils
import random
import time
import cv2
from PIL import Image
import models.layers
import addons.trees as trees
from models.vision import HTCNN, HTCNN_M, HTCNN_M_IN, LeNet5, AlexNet, AlexNet32, AlexNet32_B, AlexNet32_C, AlexNet32_D
import argparse
import threading
from time import sleep
from torchviz import make_dot, make_dot_from_trace
from addons import amath

__all__ = ("error", "LockType", "start_new_thread", "interrupt_main", "exit", "allocate_lock", "get_ident", "stack_size", "acquire", "release", "locked")


# -

class DataThread (threading.Thread):
    def __init__(self, indices, img_paths, preprocess_1 = None, preprocess_2 = None):
        threading.Thread.__init__(self)
        self.indices = indices
        self.img_paths = img_paths
        self.preprocess_1 = preprocess_1
        self.preprocess_2 = preprocess_2
        
    def run(self):
        processData(self.indices, self.img_paths, self.preprocess_1, self.preprocess_2)


def loadData(data_path, data_file):
    output = []
    with open(data_file, 'r') as f:
        for ln in f:
            fields = ln.rstrip('\n').split(',')
            output.append([os.path.join(data_path,fields[0]), int(fields[1])])
    return output

def processData(indices, img_paths, preprocess_1 = None, preprocess_2 = None):
    global tmp_output
    for idx in indices:
        p_data = Image.open(img_paths[idx])
        if preprocess_1 is not None:
            p_data = preprocess_1(p_data)
        if preprocess_2 is not None:
            p_data = preprocess_2(p_data)
        tmp_output[idx] = p_data.unsqueeze(0)


def loadInBatch(ds, r = 0, batchsize = 16, shuffle=False, preprocessor=None, im_size=None, onehot=False):
    output_data = None
    aux_labels = []
    fine_labels = None
    i = 0
    ndata = len(ds)
    hasDone = False
    im_width = im_size[2]
    im_height = im_size[1]
    im_ch = im_size[0]
    output_data = torch.zeros(batchsize, im_ch, im_height, im_width, device=device)
    while i<batchsize:
        data_rec = ds[r][0]
        img_data = None
        data_blob = None
        
        #img_data = Image.open(data_rec)
        #data_blob = preprocessor(img_data).unsqueeze(0)
        img_data = cv2.imread(data_rec)
        data_blob = preprocessor(image=img_data)['image'].unsqueeze(0)
        base_label = ds[r][1] 
        output_data[i, ...] = data_blob
        if aux_labels == []:
            j = 0
            for lv in lookup_lv_list:
                if onehot:
                    output_label = torch.zeros(batchsize, coarst_dims[j], device=device).long()
                else:
                    output_label = torch.zeros(batchsize, device=device).long()
                aux_labels.append(output_label)
                j += 1
        if fine_labels is None:
            if onehot:
                fine_labels = torch.zeros(batchsize, n_fine, device=device).long()
            else:
                fine_labels = torch.zeros(batchsize, device=device).long()
        j = 0
        for lv in lookup_lv_list:
            up_cls = lookupParent(classTree, base_label, lv)
            if onehot:
                aux_labels[j].data[i, up_cls] = 1
            else:
                aux_labels[j].data[i] = up_cls
            j += 1
        if onehot:
            fine_labels.data[i, base_label] = 1
        else:
            fine_labels.data[i] = base_label
        r += 1
        if r >= ndata:
            r = 0
            hasDone = True
            if shuffle:
                random.shuffle(ds)
        i += 1
        
    output_data = output_data.to(device)
    return output_data, aux_labels, fine_labels, r, hasDone

def loadInBatch_hp(ds, r = 0, batchsize = 16, shuffle=False, preprocessor=None, im_size=None, nThread=2):
    global tmp_output
    output_data = None
    aux_labels = []
    fine_labels = None
    i = 0
    ndata = len(ds)
    hasDone = False
    tmp_output = []
    img_paths = []
    
    im_width = im_size[2]
    im_height = im_size[1]
    im_ch = im_size[0]
    output_data = torch.zeros(batchsize, im_ch, im_height, im_width, device=device)
    while i<batchsize:
        tmp_output.append(None)
        i+=1
    i = 0
    while i<batchsize:
        data_rec = ds[r][0]
        img_data = None
        data_blob = None
        #_thread.start_new_thread(processData, (i, data_rec, preprocessor, None))
        img_paths.append(data_rec)
        base_label = ds[r][1] 
        if aux_labels == []:
            j = 0
            for lv in lookup_lv_list:
                output_label = torch.zeros(batchsize, coarst_dims[j]).long().to(device)
                output_label.require_grad = False
                aux_labels.append(output_label)
                j += 1
        if fine_labels is None:
            fine_labels = torch.zeros(batchsize, n_fine).long().to(device)
        j = 0
        for lv in lookup_lv_list:
            up_cls = lookupParent(classTree, base_label, lv)
            aux_labels[j].data[i, up_cls] = 1
            j += 1
        fine_labels.data[i, base_label] = 1
        r += 1
        if r >= ndata:
            r = 0
            hasDone = True
            if shuffle:
                random.shuffle(ds)
        i += 1
    allDone = False
    idx_i = 0
    thd = []
    n_index = int(np.ceil(float(batchsize)/nThread))
    for ni in range(nThread):
        targets = []
        for ii in range(n_index):
            targets.append(idx_i)
            idx_i+=1
            if idx_i>=batchsize:
                break
        thd.append(DataThread(targets, img_paths, preprocessor, None))
        thd[ni].start()
    for ithd in thd:
        ithd.join()
    thd = []
    i = 0
    while i<batchsize:
        output_data[i, ...] = tmp_output[i]
        i+=1
    tmp_output = []
    #output_data.require_grad = False
    fine_labels.require_grad = False
    return output_data, aux_labels, fine_labels, r, hasDone


def loadInBatch_mblob(ds, r = 0, batchsize = 16, shuffle=False, preprocessors=None, im_sizes=None,
                     general_preprocess = None):
    output_data = []
    aux_labels = []
    fine_labels = None
    i = 0
    n_output = 0
    if preprocessors is not None:
        n_output = len(preprocessors)
    else:
        n_output = len(im_sizes)
    ndata = len(ds)
    hasDone = False
    if preprocessors is None:
        raise Exception('Preprocessors cannot be empty.')
    while i<batchsize:
        data_rec = ds[r][0]
        img_data = None
        data_blob = None
        img_data = Image.open(data_rec)
        base_label = ds[r][1] 
        if general_preprocess is not None:
            img_data = general_preprocess(img_data)
        for i_output in range(n_output):
            im_width = im_sizes[i_output][2]
            im_height = im_sizes[i_output][1]
            im_ch = im_sizes[i_output][0]
            local_output_data = None
            data_blob = preprocessors[i_output](img_data).unsqueeze(0)
            
            if output_data != [] and len(output_data)>i_output:
                local_output_data = output_data[i_output]
            else:
                local_output_data = torch.zeros(batchsize, im_ch, im_height, im_width, device=device)
                output_data.append(local_output_data)
            local_output_data[i, ...] = data_blob
        if aux_labels == []:
            j = 0
            for lv in lookup_lv_list:
                output_label = torch.zeros(batchsize, coarst_dims[j]).long().to(device)
                output_label.require_grad = False
                aux_labels.append(output_label)
                j += 1
        if fine_labels is None:
            fine_labels = torch.zeros(batchsize, n_fine).long().to(device)
        j = 0
        for lv in lookup_lv_list:
            up_cls = lookupParent(classTree, base_label, lv)
            aux_labels[j].data[i, up_cls] = 1
            j += 1
        fine_labels.data[i, base_label] = 1
        r += 1
        if r >= ndata:
            r = 0
            hasDone = True
            if shuffle:
                random.shuffle(ds)
        i += 1
        
    
    fine_labels.require_grad = False
    return output_data, aux_labels, fine_labels, r, hasDone

def lookupParent(tree, fine_node, upper_lv=1):
    return tree[fine_node][upper_lv-1]

def accumulateList(list1, list2):
    output = []
    for i in range(len(list1)):
        output.append((list1[i] + list2[i]) * 0.5)
    return output

def computeBatchAccuracy(pred, expected, onehot=False):
    output = []
    n_output = len(pred)
    n_batch = pred[0].shape[0]
    for i in range(n_output):
        local_result = 0.0
        for j in range(n_batch):
            cls_pred = pred[i][j].argmax()
            if onehot:
                cls_exp = expected[i][j,...].argmax()
            else:
                cls_exp = expected[i][j]
            #print((cls_pred, cls_exp))
            if cls_pred == cls_exp:
                local_result += 1.0
        local_result /= n_batch
        output.append(local_result)
    return output

def computeAccuracy(dataset, model, batchsize = 1, withAux = False, preprocessor = None, withLoss = None):
    data_count = len(dataset)
    ptr = 0
    batch_len = int(np.floor(float(data_count)/batchsize))
    batch_elen = int(np.ceil(float(data_count)/batchsize))
    output = []
    aux_output = []
    loss_v = 0
    for i in range(batch_len):
        batch_data, expected_aux, expected_fine, ptr, _ = loadInBatch(dataset, ptr, batchsize, preprocessor=preprocessor,
                                                                        im_size = input_sizes)
        pred_final, pred_aux = model(batch_data)
        if withLoss is not None:
            v_loss = withLoss(pred_final, expected_fine).item()
            loss_v += v_loss
        batch_result = computeBatchAccuracy([pred_final], [expected_fine])
        if output == []:
            output = batch_result
        else:
            for j in range(len(output)):
                output[j] += batch_result[j]
        if withAux:
            batch_aux_result = computeBatchAccuracy(pred_aux, expected_aux + [expected_fine])
            if aux_output == []:
                aux_output = batch_aux_result
            else:
                for j in range(len(aux_output)):
                    aux_output[j] += batch_aux_result[j]
    if batchsize!=1 and batch_len != batch_elen:
        tmp_batchsize = data_count - ptr
        batch_data, expected_aux, expected_fine, ptr, _ = loadInBatch(dataset, ptr, tmp_batchsize, preprocessor=preprocessor,
                                                                        im_size = input_sizes)
        pred_final, pred_aux = model(batch_data)
        batch_result = computeBatchAccuracy([pred_final], [expected_fine])
        for j in range(len(output)):
            output[j] += batch_result[j]
            output[j] /= batch_len + 1
        if withAux:
            batch_aux_result = computeBatchAccuracy(pred_aux, expected_aux + [expected_fine])
            for j in range(len(aux_output)):
                aux_output[j] += batch_aux_result[j]
                aux_output[j] /= batch_len + 1
    else:
        print('damn')
        for j in range(len(output)):
            output[j] /= batch_len
        if withAux:
            for j in range(len(aux_output)):
                aux_output[j] /= batch_len
        
    return output, aux_output, loss_v/batch_elen

def computeAccuracy_m_in(dataset, model, batchsize = 1, withAux = False, preprocessors = None, im_sizes = None):
    data_count = len(dataset)
    ptr = 0
    batch_len = int(np.floor(float(data_count)/batchsize))
    batch_elen = int(np.ceil(float(data_count)/batchsize))
    output = []
    aux_output = []
    for i in range(batch_len):
        batch_data, expected_aux, expected_fine, ptr, _ = loadInBatch_mblob(dataset, ptr, batchsize, preprocessors=preprocessors, im_sizes=im_sizes)
        pred_final, pred_aux = model(batch_data)
        batch_result = computeBatchAccuracy([pred_final], [expected_fine])
        if output == []:
            output = batch_result
        else:
            for j in range(len(output)):
                output[j] += batch_result[j]
        if withAux:
            batch_aux_result = computeBatchAccuracy(pred_aux, expected_aux + [expected_fine])
            if aux_output == []:
                aux_output = batch_aux_result
            else:
                for j in range(len(aux_output)):
                    aux_output[j] += batch_aux_result[j]
    if batchsize!=1 and batch_len != batch_elen:
        tmp_batchsize = data_count - ptr
        batch_data, expected_aux, expected_fine, ptr, _ = loadInBatch_mblob(dataset, ptr, tmp_batchsize, preprocessors=preprocessors, im_sizes=im_sizes)
        pred_final, pred_aux = model(batch_data)
        batch_result = computeBatchAccuracy([pred_final], [expected_fine])
        for j in range(len(output)):
            output[j] += batch_result[j]
            output[j] /= batch_len + 1
        if withAux:
            batch_aux_result = computeBatchAccuracy(pred_aux, expected_aux + [expected_fine])
            for j in range(len(aux_output)):
                aux_output[j] += batch_aux_result[j]
                aux_output[j] /= batch_len + 1
    else:
        for j in range(len(output)):
            output[j] /= batch_len
        if withAux:
            for j in range(len(aux_output)):
                aux_output[j] /= batch_len
        
    return output, aux_output

def compute_loss_diff(losses, absolute=False, losses_name=None):
    output = []
    output_names = []
    n_loss = len(losses)
    #n_output = amath.ncr(n_loss, r=2)
    for i in range(n_loss):
        for j in range(i, n_loss):
            if i==j:
                continue
            y = losses[j]-losses[i]
            if losses_name is not None:
                output_names.append('%s - %s'%(losses_name[j], losses_name[i]))
            if absolute:
                y = np.abs(y)
            output.append(y)
    return output, output_names


def train(trainset, valset, label_file, output_path, output_fname, 
          start_lr=0.1, lr_discount=0.1, lr_steps=[], epoch=30,
          train_batch = 16, val_batch = 16, val_at = 10,
          checkpoint = None, jud_at = -1, aux_scaler = 0.3, final_scaler = 1.0, fine_scaler = 1.0,
          preprocessor = None, v_preprocessor = None, isConditionProb=True, coastBack=True,
          f_weights = None
         ):
    global backbone
    best_v_result = 0.0
    model = HTCNN(label_file, with_aux = True, with_fc = True, backbone=backbone,
              isCuda=True, isConditionProb=isConditionProb, coastBack=coastBack, weights=f_weights, autosizeFC = True).cuda()
    
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name)
    
    output_filepath = os.path.join(output_path, output_fname)

    if checkpoint is not None and os.path.isfile(checkpoint):
        
        model_params = torch.load(checkpoint)
        model.load_state_dict(model_params)
        backbone = model.backbone_nn
        print('Loaded from checkpoint %s'%checkpoint)
    
    #sample, _, _, _, _ = loadInBatch(trainset, batchsize = 1)
    #writer.add_graph(model, sample)
    #writer.close()
    
    v_result = 0
    
    model.eval()
    with torch.no_grad():
        val_result, aux_val_result, v_loss = computeAccuracy(valset, model, val_batch, withAux=True, preprocessor = v_preprocessor)
        v_result = val_result[0]
        print('Validation Accuracy: %f'%v_result)
        print(aux_val_result)
        best_v_result = v_result
    
    lr = start_lr
    
    param_list = [{'params':model.proj_layers.parameters()}, 
                  {'params':model.fc_s.parameters()},
                  {'params':model.backbone_nn.parameters()}
                 ]
    
    optimizer = optim.SGD(param_list, lr=lr, momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.Adagrad(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_discount)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = int(np.ceil(4.0/val_at)), threshold=1e-3)
    
    # create losses
    losses = []
    aux_loss_names = []
    aux_val_names = []
    final_loss = nn.NLLLoss()
    for lv in lookup_lv_list:
        losses.append(nn.NLLLoss())
        aux_loss_names.append('Coarst %d loss'%lv)
        aux_val_names.append('Level %d accuracy'%lv)
    losses.append(nn.NLLLoss())
    aux_loss_names.append('Fine loss')
    aux_val_names.append('Fine accuracy')
    n_aux = len(losses) - 1
    aux_accuracy = {}
    vloss = nn.NLLLoss()
    
    for i in range(epoch):
        # training phase
        model.train()
        ptr = 0
        hasFinishEpoch = False
        epoch_result = []
        epoch_aux_losses_v = []
        epoch_loss_v = 0
        iter_c = 0
        avg_model_fwd_elapsed_time = 0.0
        while not hasFinishEpoch:
            optimizer.zero_grad()
            
            pp_start_time = time.time()
            batch_input, gt_aux, gt_final, ptr, hasFinishEpoch = loadInBatch(trainset, ptr, train_batch, shuffle=True,
                                                                            preprocessor=preprocessor, im_size = input_sizes)
            pp_elapsed_time = time.time() - pp_start_time
            
            model_start_time = time.time()
            pred_final, pred_aux = model(batch_input)
            model_fwd_elapsed_time = time.time() - model_start_time
            avg_model_fwd_elapsed_time = (avg_model_fwd_elapsed_time + model_fwd_elapsed_time) / 2.0
            
            iloss = 0
            fine_loss = losses[-1](pred_aux[-1], gt_final)*fine_scaler
            total_loss = fine_loss
            for i_aux in range(n_aux):
                aux_loss = losses[i_aux](pred_aux[i_aux], gt_aux[i_aux])
                total_loss = total_loss + aux_loss * aux_scaler
                #total_loss = torch.sum(aux_loss)
                aux_loss_v = aux_loss.item()
                if epoch_aux_losses_v == []:
                    epoch_aux_losses_v.append(aux_loss_v)
                else:
                    epoch_aux_losses_v[iloss] += aux_loss_v
                iloss += 1
            
            f_loss = final_loss(pred_final, gt_final)
            total_loss = total_loss + f_loss * final_scaler
            fine_loss_v = fine_loss.item()
            if len(epoch_aux_losses_v) <= iloss:
                epoch_aux_losses_v.append(fine_loss_v)
            else:
                epoch_aux_losses_v[iloss] += fine_loss_v
            # compute gradients
            total_loss.backward()
            
            # update weights
            optimizer.step()
            
            if iter_c == 0:
                epoch_loss_v = total_loss.item()
            else:
                epoch_loss_v += total_loss.item()
            
            if epoch_loss_v == 0:
                epoch_loss_v = total_loss.item()
            
            result = computeBatchAccuracy([pred_final.data],[gt_final])
            if epoch_result == []:
                epoch_result = result
            else:
                epoch_result = accumulateList(epoch_result, result)
            iter_c += 1
            print('[iteration %d]Data Loading Time:%f seconds; Computation Time:%f seconds'%(iter_c,pp_elapsed_time, model_fwd_elapsed_time))
        
        #scheduler.step()
        plot_loss = {}
        for iloss in range(n_aux+1):
            epoch_aux_losses_v[iloss] /= iter_c
            plot_loss[aux_loss_names[iloss]] = epoch_aux_losses_v[iloss]
            plotter.plot('loss', 'aux %d'%iloss,'Losses', i, epoch_aux_losses_v[iloss])
            #print('%s: %f, '%(aux_loss_names[iloss], epoch_aux_losses_v[iloss]), end='')
        epoch_loss_v /= iter_c
        #lot_loss['total loss'] = epoch_loss_v
        plotter.plot('loss', 'total','Total Loss', i, epoch_loss_v)
        #writer.add_scalars('training loss', 
        #                  plot_loss,
        #                  i)
        #print('Fine loss: %f'%epoch_loss_v)
        print(plot_loss)
        
        # validation phase
        if i % val_at == 0 or (i+1)==epoch:
            disp_i = i+1
            if i==0:
                disp_i = 0
            print('Validating...')
            model.eval()
            with torch.no_grad():
                val_result, aux_val_result, v_loss = computeAccuracy(valset, model, val_batch, withAux=True, preprocessor = v_preprocessor,
                                                                    withLoss = vloss)
                for iacc in range(len(aux_val_names)):
                    aux_accuracy[aux_val_names[iacc]] = aux_val_result[iacc]
                v_result = val_result[0]
                print('Validation Accuracy: %f'%v_result)
                print(aux_accuracy)
                if v_result > best_v_result:
                    print('Best model found and saving it.')
                    torch.save(model.state_dict(), output_filepath)
                    best_v_result = v_result
                plotter.plot('Validation Loss', 'final','Validation Loss', disp_i, v_loss)
                scheduler.step(v_loss)
        #if i in lr_steps:
        #    olr = lr
        #    lr *= lr_discount
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = lr
        #    print('learning rate has been discounted from %f to %f'%(olr, lr))
            for i_aux in range(len(aux_accuracy)):
                plotter.plot('acc','aux %d'%(i_aux),'Accuracy', disp_i, aux_accuracy[aux_val_names[i_aux]])
            plotter.plot('acc','final','Final Accuracy', disp_i, v_result)
        #writer.add_scalars('Auxiliary Accuracy', 
        #                  aux_accuracy,
        #                  i)
        #writer.add_scalar('Final Accuracy', 
        #                  v_result,
        #                  i)
            
    print('Model has been trained.')
    model = None

def train_mb(trainset, valset, label_file, output_path, output_fname, 
          start_lr=0.1, lr_discount=0.1, lr_steps=[], epoch=30,
          train_batch = 16, val_batch = 16, val_at = 10,
          checkpoint = None, jud_at = -1, aux_scaler = 0.3, final_scaler = 1.0, fine_scaler = 1.0,
          preprocessor = None, v_preprocessor = None, isConditionProb=True, coastBack=True,
          f_weights = None
         ):
    
    best_v_result = 0.0
    backbones = nn.ModuleList([backbone_1, backbone_2])
    
    model = HTCNN_M(label_file, with_aux = True, with_fc = False, backbones=backbones,
              isCuda=True, isConditionProb=isConditionProb, coastBack=coastBack, weights=f_weights).to(device)
    
    #with torch.onnx.set_training(model, False):
        #trace, _ = torch.jit.get_trace_graph(model, args=(x,))
        #dot = make_dot_from_trace(trace)
        #dot.format = 'png'
        #dot.render(os.path.join(output_path, 'model.png'))
    
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name)
    
    output_filepath = os.path.join(output_path, output_fname)

    if checkpoint is not None and os.path.isfile(checkpoint):
        model_params = torch.load(checkpoint)
        model.load_state_dict(model_params)
        backbones = model.backbones
        print('Loaded from checkpoint %s'%checkpoint)
    
    #sample, _, _, _, _ = loadInBatch(trainset, batchsize = 1)
    #writer.add_graph(model, sample)
    #writer.close()
    
    
    v_result = 0
    f_v_result = 0
    
    disp_x = torch.zeros(1, backbone_1.input_dim[0], backbone_1.input_dim[1], backbone_1.input_dim[2], device=device)
    disp, _ = model(disp_x)
    dot = make_dot(disp, params = dict(model.named_parameters()))
    dot.render(os.path.join(output_path, "model.png"))
    #torch.onnx.export(model, disp_x, os.path.join(output_path, "model.onnx"), input_names=['X'], output_names=['Y'], opset_version=11)
    disp_x = None
    
    model.eval()
    with torch.no_grad():
        
        val_result, aux_val_result, v_loss = computeAccuracy(valset, model, val_batch, withAux=True, preprocessor = v_preprocessor)
        v_result = val_result[0]
        print('Validation Accuracy: %f'%v_result)
        print(aux_val_result)
        best_v_result = v_result
    
    
    lr = start_lr
    
    param_list = [{'params':model.proj_layers.parameters()}, 
                  {'params':model.fc_s.parameters()},
                  {'params':model.backbones[0].parameters()}
                 ]
    
    fine_lr = start_lr
    fine_lr_rate = lr_discount
    fine_steps = lr_steps
    if args.f_same!=1:
        fine_lr = args.f_lr
        fine_lr_rate = args.f_lr_discount
        fine_steps = [int(istep) for istep in args.f_step_down.split(',')]
    
    default_rated_scaler = args.lr
    rated_scaler = default_rated_scaler
    for i in range(1, len(backbones)):
        param_list.append({'params':model.backbones[i].parameters(), 'lr':rated_scaler})
    
    optimizer = optim.SGD(param_list, lr=lr, momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.ASGD(param_list, lr=lr, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=0.01)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_discount)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = int(np.ceil(4.0/val_at)), threshold=1e-3)
    
    # create losses
    losses = nn.ModuleList()
    aux_loss_names = []
    aux_val_names = []
    final_loss = nn.NLLLoss(size_average=None, reduce=None)
    for lv in lookup_lv_list:
        losses.append(nn.NLLLoss(size_average=None, reduce=None))
        aux_loss_names.append('Coarst %d loss'%lv)
        aux_val_names.append('Level %d accuracy'%lv)
    losses.append(nn.NLLLoss(size_average=None, reduce=None))
    aux_loss_names.append('Fine loss')
    aux_val_names.append('Fine accuracy')
    n_aux = len(losses) - 1
    aux_accuracy = {}
    vloss = nn.NLLLoss(size_average=None, reduce=None)
    
    ptr = 0
    for i in range(epoch):
        # training phase
        model.train()
        hasFinishEpoch = False
        epoch_result = []
        epoch_aux_losses_v = []
        epoch_final_loss_v = 0
        epoch_loss_v = 0
        iter_c = 0
        avg_model_fwd_elapsed_time = 0.0
        while not hasFinishEpoch:
            
            pp_start_time = time.time()
            batch_input, gt_aux, gt_final, ptr, hasFinishEpoch = loadInBatch(trainset, ptr, train_batch, shuffle=True,
                                                                            preprocessor=preprocessor, im_size = input_sizes)
            pp_elapsed_time = time.time() - pp_start_time
            
            model_start_time = time.time()
            pred_final, pred_aux = model(batch_input)
            model_fwd_elapsed_time = time.time() - model_start_time
            avg_model_fwd_elapsed_time = (avg_model_fwd_elapsed_time + model_fwd_elapsed_time) / 2.0
            
            iloss = 0
            
            aux_loss_vlist = []
            
            fine_loss = losses[-1](pred_aux[-1], gt_final)
            aux_loss_vlist.append(fine_loss.item())
            total_loss = fine_loss*fine_scaler
            for i_aux in range(n_aux):
                aux_loss = losses[i_aux](pred_aux[i_aux], gt_aux[i_aux]) 
                total_loss = total_loss + aux_loss * aux_scaler
                aux_loss_v = aux_loss.item()
                aux_loss_vlist.append(aux_loss_v)
                if epoch_aux_losses_v == []:
                    epoch_aux_losses_v.append(aux_loss_v)
                else:
                    epoch_aux_losses_v[iloss] += aux_loss_v
                iloss += 1
            f_loss = final_loss(pred_final, gt_final)
            epoch_final_loss_v += f_loss.item()
            total_loss = total_loss + f_loss * final_scaler
            fine_loss_v = fine_loss.item()
            if len(epoch_aux_losses_v) <= iloss:
                epoch_aux_losses_v.append(fine_loss_v)
            else:
                epoch_aux_losses_v[iloss] += fine_loss_v
                
            aux_loss_str = ','.join(['%.4f'%loss_v for loss_v in aux_loss_vlist])
            
            optimizer.zero_grad()
            # compute gradients
            total_loss.backward()
            
            # update weights
            optimizer.step()
            #scheduler.step()
            
            if iter_c == 0:
                epoch_loss_v = total_loss.item()
            else:
                epoch_loss_v += total_loss.item()
            
            if epoch_loss_v == 0:
                epoch_loss_v = total_loss.item()
            
            result = computeBatchAccuracy([pred_final.data],[gt_final])
            if epoch_result == []:
                epoch_result = result
            else:
                epoch_result = accumulateList(epoch_result, result)
            iter_c += 1
            print('[iteration %d]Data Loading Time:%f seconds; Computation Time:%f seconds [loss:%f, aux loss:%s]'%(iter_c,
                                                                                                       pp_elapsed_time, 
                                                                                                       model_fwd_elapsed_time, 
                                                                                                       total_loss.item(),
                                                                                                       aux_loss_str
                                                                                                      ))
        #scheduler.step()
        #print('Training Loss:', end='')
        plot_loss = {}
        aux_losses_names = []
        for iloss in range(n_aux+1):
            epoch_aux_losses_v[iloss] /= iter_c
            plot_loss[aux_loss_names[iloss]] = epoch_aux_losses_v[iloss]
            aux_losses_names.append('aux %d'%iloss)
            plotter.plot('loss', 'aux %d'%iloss,'Losses', i, epoch_aux_losses_v[iloss])
            #print('%s: %f, '%(aux_loss_names[iloss], epoch_aux_losses_v[iloss]), end='')
        epoch_loss_v /= iter_c
        epoch_final_loss_v /=iter_c
        #lot_loss['total loss'] = epoch_loss_v
        plotter.plot('loss', 'final','Losses', i, epoch_final_loss_v)
        plotter.plot('loss', 'total','Losses', i, epoch_loss_v)
        
        #writer.add_scalars('training loss', 
        #                  plot_loss,
        #                  i)
        #print('Fine loss: %f'%epoch_loss_v)
        print(plot_loss)
        
        last_top1 = f_v_result
        # validation phase
        if i % val_at == 0 or (i+1)==epoch:
            disp_i = i+1
            if i==0:
                disp_i = 0
            print('Validating...')
            model.eval()
            with torch.no_grad():
                val_result, aux_val_result, v_loss = computeAccuracy(valset, model, val_batch, withAux=True, preprocessor = v_preprocessor,
                                                                    withLoss = vloss)
                for iacc in range(len(aux_val_names)):
                    aux_accuracy[aux_val_names[iacc]] = aux_val_result[iacc]
                f_v_result = aux_val_result[-1]
                v_result = val_result[0]
                print('Validation Accuracy: %f'%v_result)
                print(aux_accuracy)
                if v_result > best_v_result:
                    print('Best model found and saving it.')
                    torch.save(model.state_dict(), output_filepath)
                    best_v_result = v_result
                plotter.plot('Validation Loss', 'final','Validation Loss', disp_i, v_loss)
                scheduler.step(v_loss)
            for i_aux in range(len(aux_accuracy)):
                plotter.plot('acc','aux %d'%(i_aux),'Accuracy', disp_i, aux_accuracy[aux_val_names[i_aux]])
                plotter.plot('acc','final','Final Accuracy', disp_i, v_result)
        
        #diff_losses, diff_names = compute_loss_diff(epoch_aux_losses_v, losses_name = aux_losses_names)
        #for i_dloss in range(len(diff_losses)):
        #    if f_v_result-last_top1>0.0:
        #        plotter.plot_scatter('difference of loss', '%s (+)'%diff_names[i_dloss], 'Different of Loss', i, diff_losses[i_dloss],
        #                            [0,255,0])
        #    elif f_v_result-last_top1==0.0:
        #        plotter.plot_scatter('difference of loss', '%s (*)'%diff_names[i_dloss], 'Different of Loss', i, diff_losses[i_dloss],
        #                            [10,10,10])
        #    else:
        #        plotter.plot_scatter('difference of loss', '%s (-)'%diff_names[i_dloss], 'Different of Loss', i, diff_losses[i_dloss],
        #                            [255,0,0])
        
        if args.f_same == 1:
            
            #old_lr = scheduler.get_last_lr()
            #scheduler.step()
            #print('learning rate has been discounted from %f to %f'%(old_lr, scheduler.get_last_lr()))
            if (i+1) in lr_steps:
                pass
                #olr = lr
                #lr *= lr_discount
                #for param_group in optimizer.param_groups:
                #    param_group['lr'] = lr
                #print('learning rate has been discounted from %f to %f'%(olr, lr))
        else:
            if (i+1) in lr_steps:
                olr = lr
                lr *= lr_discount
                lr_t = len(param_list)
                lr_i = 0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    lr_i += 1
                    if lr_i>=(lr_t-1):
                        break
                print('learning rate has been discounted from %f to %f for aux'%(olr, lr))
            if (i+1) in fine_steps:
                olr = fine_lr
                lr *= fine_lr_rate
                lr_i = 0
                lr_t = len(param_list)
                for param_group in optimizer.param_groups:
                    lr_i += 1
                    if lr_i!=lr_t:
                        continue
                    param_group['lr'] = lr
                print('learning rate has been discounted from %f to %f for fine'%(olr, lr))
                
        
        #writer.add_scalars('Auxiliary Accuracy', 
        #                  aux_accuracy,
        #                  i)
        #writer.add_scalar('Final Accuracy', 
        #                  v_result,
        #                  i)
            
    print('Model has been trained.')
    model = None

def train_mb_in(trainset, valset, label_file, output_path, output_fname, 
          start_lr=0.1, lr_discount=0.1, lr_steps=[], epoch=30,
          train_batch = 16, val_batch = 16, val_at = 10,
          checkpoint = None, jud_at = -1, aux_scaler = 0.3, final_scaler = 1.0, fine_scaler = 1.0,
          preprocessor = None, v_preprocessor = None, im_size = None, general_process = None, isConditionProb=True, coastBack=True,
          f_weights = None
         ):
    
    best_v_result = 0.0
    backbones = nn.ModuleList([backbone_1, backbone_2])
    model = HTCNN_M_IN(label_file, with_aux = True, with_fc = False, backbones=backbones,
              isCuda=True, isConditionProb=isConditionProb, coastBack=coastBack, weights=f_weights)
    
    model = nn.DataParallel(model).cuda()
    output_filepath = os.path.join(output_path, output_fname)

    if checkpoint is not None and os.path.isfile(checkpoint):
        model_params = torch.load(checkpoint)
        model.load_state_dict(model_params)
        backbones = model.backbones
        print('Loaded from checkpoint %s'%checkpoint)
    
    
    v_result = 0
    
    model.eval()
    with torch.no_grad():
        val_result, aux_val_result, v_loss = computeAccuracy_m_in(valset, model, val_batch, withAux=True,
                                                         im_sizes = im_size,
                                                         preprocessors = v_preprocessor)
        v_result = val_result[0]
        print('Validation Accuracy: %f'%v_result)
        print(aux_val_result)
        best_v_result = v_result
    
    
    lr = start_lr
    optimizer = optim.SGD(param_list, lr=lr, momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.Adagrad(model.parameters(), lr=lr)
    
    
    # create losses
    losses = []
    aux_loss_names = []
    aux_val_names = []
    final_loss = nn.CrossEntropyLoss()
    for lv in lookup_lv_list:
        losses.append(nn.CrossEntropyLoss())
        aux_loss_names.append('Coarst %d loss'%lv)
        aux_val_names.append('Level %d accuracy'%lv)
    losses.append(nn.CrossEntropyLoss())
    aux_loss_names.append('Fine loss')
    aux_val_names.append('Fine accuracy')
    n_aux = len(losses) - 1
    aux_accuracy = {}
    
    for i in range(epoch):
        # training phase
        model.train()
        ptr = 0
        hasFinishEpoch = False
        epoch_result = []
        epoch_aux_losses_v = []
        epoch_loss_v = 0
        iter_c = 0
        avg_model_fwd_elapsed_time = 0.0
        
        blobs = []
        
        while not hasFinishEpoch:
            optimizer.zero_grad()
            
            pp_start_time = time.time()
            batch_input, gt_aux, gt_final, ptr, hasFinishEpoch = loadInBatch_mblob(trainset, ptr, train_batch, shuffle=True,
                                                                            preprocessors=preprocessor,
                                                                                  im_sizes=im_size,
                                                                                  general_preprocess=general_process)
            pp_elapsed_time = time.time() - pp_start_time
            
            
            model_start_time = time.time()
            pred_final, pred_aux = model(batch_input)
            model_fwd_elapsed_time = time.time() - model_start_time
            avg_model_fwd_elapsed_time = (avg_model_fwd_elapsed_time + model_fwd_elapsed_time) / 2.0
            
            iloss = 0
            fine_loss = losses[-1](pred_aux[-1], gt_final)*fine_scaler
            total_loss = fine_loss
            for i_aux in range(n_aux):
                aux_loss = losses[i_aux](pred_aux[i_aux], gt_aux[i_aux])
                total_loss = total_loss + aux_loss * aux_scaler
                aux_loss_v = aux_loss.item()
                if epoch_aux_losses_v == []:
                    epoch_aux_losses_v.append(aux_loss_v)
                else:
                    epoch_aux_losses_v[iloss] += aux_loss_v
                iloss += 1
            f_loss = final_loss(pred_final, gt_final)
            total_loss = total_loss + f_loss* final_scaler
            fine_loss_v = fine_loss.item()
            if len(epoch_aux_losses_v) <= iloss:
                epoch_aux_losses_v.append(fine_loss_v)
            else:
                epoch_aux_losses_v[iloss] += fine_loss_v
            # compute gradients
            total_loss.backward()
            
            # update weights
            optimizer.step()
            
            if iter_c == 0:
                epoch_loss_v = total_loss.item()
            else:
                epoch_loss_v += total_loss.item()
            
            if epoch_loss_v == 0:
                epoch_loss_v = total_loss.item()
            
            result = computeBatchAccuracy([pred_final],[gt_final])
            if epoch_result == []:
                epoch_result = result
            else:
                epoch_result = accumulateList(epoch_result, result)
            iter_c += 1
            #if iter_c == 1:
            print('[iteration %d]Data Loading Time:%f seconds; Computation Time:%f seconds'%(iter_c,pp_elapsed_time, model_fwd_elapsed_time))
        
        plot_loss = {}
        for iloss in range(n_aux+1):
            epoch_aux_losses_v[iloss] /= iter_c
            plot_loss[aux_loss_names[iloss]] = epoch_aux_losses_v[iloss]
            plotter.plot('loss', 'aux %d'%iloss,'Losses', i, epoch_aux_losses_v[iloss])
        epoch_loss_v /= iter_c
        plotter.plot('loss', 'total','Total Loss', i, epoch_loss_v)
        print(plot_loss)
        
        # validation phase
        if i % val_at == 0:
            print('Validating...')
            model.eval()
            with torch.no_grad():
                val_result, aux_val_result, v_loss = computeAccuracy_m_in(valset, model, val_batch, withAux=True,
                                                                 im_sizes = im_size,
                                                         preprocessors = v_preprocessor)
                for iacc in range(len(aux_val_names)):
                    aux_accuracy[aux_val_names[iacc]] = aux_val_result[iacc]
                v_result = val_result[0]
                print('Validation Accuracy: %f'%v_result)
                print(aux_accuracy)
                if v_result > best_v_result:
                    print('Best model found and saving it.')
                    torch.save(model.state_dict(), output_filepath)
                    best_v_result = v_result
        if i in lr_steps:
            olr = lr
            lr *= lr_discount
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('learning rate has been discounted from %f to %f'%(olr, lr))
        for i_aux in range(len(aux_accuracy)):
            plotter.plot('acc','aux %d'%(i_aux),'Accuracy', i, aux_accuracy[aux_val_names[i_aux]])
        plotter.plot('acc','final','Final Accuracy', i, v_result)
        
    print('Model has been trained.')
    model = None

def train_coast(trainset, valset, label_file, output_path, output_fname, 
          start_lr=0.1, lr_discount=0.1, lr_steps=[], epoch=30,
          train_batch = 16, val_batch = 16, val_at = 10, n_coast=0,
          checkpoint = None, jud_at = -1, aux_scaler = 0.3, final_scaler = 1.0, fine_scaler = 1.0,
          preprocessor = None, v_preprocessor = None, isConditionProb=True, coastBack=True,
          f_weights = None
         ):
    global backbone
    best_v_result = 0.0
    model = HTCNN(label_file, with_aux = True, with_fc = True, backbone=backbone,
              isCuda=True, isConditionProb=isConditionProb, coastBack=coastBack, weights=f_weights).cuda()
    
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name)
    
    output_filepath = os.path.join(output_path, output_fname)

    if checkpoint is not None and os.path.isfile(checkpoint):
        
        backbone.load_state_dict(torch.load(checkpoint), strict=False)
        model.load_state_dict(torch.load(checkpoint), strict=False)
        print('Loaded from checkpoint %s'%checkpoint)
    
    #sample, _, _, _, _ = loadInBatch(trainset, batchsize = 1)
    #writer.add_graph(model, sample)
    #writer.close()
    
    v_result = 0
    
    backbone.eval()
    model.eval()
    with torch.no_grad():
        val_result, aux_val_result, v_loss = computeAccuracy(valset, model, val_batch, withAux=True, preprocessor = v_preprocessor)
        v_result = val_result[0]
        print('Validation Accuracy: %f'%v_result)
        print(aux_val_result)
        best_v_result = v_result
    
    lr = start_lr
    optimizer = optim.SGD(model.parameters(), lr=lr)
    #optimizer = optim.Adagrad(model.parameters(), lr=lr)
    
    
    # create losses
    losses = []
    aux_loss_names = []
    aux_val_names = []
    final_loss = nn.MultiLabelSoftMarginLoss()
    for lv in lookup_lv_list:
        losses.append(nn.MultiLabelSoftMarginLoss())
        aux_loss_names.append('Coarst %d loss'%lv)
        aux_val_names.append('Level %d accuracy'%lv)
    losses.append(nn.MultiLabelSoftMarginLoss())
    aux_loss_names.append('Fine loss')
    aux_val_names.append('Fine accuracy')
    n_aux = len(losses) - 1
    aux_accuracy = {}
    
    for i in range(epoch):
        # training phase
        backbone.train()
        model.train()
        ptr = 0
        hasFinishEpoch = False
        epoch_result = []
        epoch_aux_losses_v = []
        epoch_loss_v = 0
        iter_c = 0
        avg_model_fwd_elapsed_time = 0.0
        while not hasFinishEpoch:
            optimizer.zero_grad()
            
            pp_start_time = time.time()
            batch_input, gt_aux, gt_final, ptr, hasFinishEpoch = loadInBatch(trainset, ptr, train_batch, shuffle=True,
                                                                            preprocessor=preprocessor, im_size = input_sizes)
            pp_elapsed_time = time.time() - pp_start_time
            
            model_start_time = time.time()
            pred_final, pred_aux = model(batch_input)
            model_fwd_elapsed_time = time.time() - model_start_time
            avg_model_fwd_elapsed_time = (avg_model_fwd_elapsed_time + model_fwd_elapsed_time) / 2.0
            
            iloss = 0
            fine_loss = losses[-1](pred_aux[-1], gt_final)*fine_scaler
            total_loss = fine_loss
            for i_aux in range(n_aux):
                aux_loss = losses[i_aux](pred_aux[i_aux], gt_aux[i_aux]) 
                total_loss = total_loss + aux_loss * aux_scaler
                #total_loss = torch.sum(aux_loss)
                aux_loss_v = aux_loss.item()
                if epoch_aux_losses_v == []:
                    epoch_aux_losses_v.append(aux_loss_v)
                else:
                    epoch_aux_losses_v[iloss] += aux_loss_v
                iloss += 1
            
            f_loss = final_loss(pred_final, gt_final)
            total_loss = total_loss + f_loss * final_scaler
            fine_loss_v = fine_loss.item()
            if len(epoch_aux_losses_v) <= iloss:
                epoch_aux_losses_v.append(fine_loss_v)
            else:
                epoch_aux_losses_v[iloss] += fine_loss_v
            # compute gradients
            total_loss.backward()
            
            # update weights
            optimizer.step()
            
            if iter_c == 0:
                epoch_loss_v = total_loss.item()
            else:
                epoch_loss_v += total_loss.item()
            
            if epoch_loss_v == 0:
                epoch_loss_v = total_loss.item()
            
            result = computeBatchAccuracy([pred_final],[gt_final])
            if epoch_result == []:
                epoch_result = result
            else:
                epoch_result = accumulateList(epoch_result, result)
            iter_c += 1
            print('[iteration %d]Data Loading Time:%f seconds; Computation Time:%f seconds'%(iter_c,pp_elapsed_time, model_fwd_elapsed_time))
        
        #print('Training Loss:', end='')
        #print('Training Loss:', end='')
        plot_loss = {}
        for iloss in range(n_aux+1):
            epoch_aux_losses_v[iloss] /= iter_c
            plot_loss[aux_loss_names[iloss]] = epoch_aux_losses_v[iloss]
            plotter.plot('loss', 'aux %d'%iloss,'Losses', i, epoch_aux_losses_v[iloss])
            #print('%s: %f, '%(aux_loss_names[iloss], epoch_aux_losses_v[iloss]), end='')
        epoch_loss_v /= iter_c
        #lot_loss['total loss'] = epoch_loss_v
        plotter.plot('loss', 'total','Total Loss', i, epoch_loss_v)
        #writer.add_scalars('training loss', 
        #                  plot_loss,
        #                  i)
        #print('Fine loss: %f'%epoch_loss_v)
        print(plot_loss)
        
        # validation phase
        if i % val_at == 0:
            print('Validating...')
            backbone.eval()
            model.eval()
            with torch.no_grad():
                val_result, aux_val_result, v_loss = computeAccuracy(valset, model, val_batch, withAux=True, preprocessor = v_preprocessor)
                for iacc in range(len(aux_val_names)):
                    aux_accuracy[aux_val_names[iacc]] = aux_val_result[iacc]
                v_result = val_result[0]
                print('Validation Accuracy: %f'%v_result)
                print(aux_accuracy)
                if v_result > best_v_result:
                    print('Best model found and saving it.')
                    torch.save(model.state_dict(), output_filepath)
                    best_v_result = v_result
        if i in lr_steps:
            olr = lr
            lr *= lr_discount
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('learning rate has been discounted from %f to %f'%(olr, lr))
        for i_aux in range(len(aux_accuracy)):
            plotter.plot('acc','aux %d'%(i_aux),'Accuracy', i, aux_accuracy[aux_val_names[i_aux]])
        plotter.plot('acc','final','Final Accuracy', i, v_result)
        #writer.add_scalars('Auxiliary Accuracy', 
        #                  aux_accuracy,
        #                  i)
        #writer.add_scalar('Final Accuracy', 
        #                  v_result,
        #                  i)
            
    print('Model has been trained.')
    model = None


def train_mb_coast(trainset, valset, label_file, output_path, output_fname, 
          start_lr=0.1, lr_discount=0.1, lr_steps=[], epoch=30,
          train_batch = 16, val_batch = 16, val_at = 10, n_coast=0,
          checkpoint = None, jud_at = -1, aux_scaler = 0.3, final_scaler = 1.0, fine_scaler = 1.0,
          preprocessor = None, v_preprocessor = None, isConditionProb=True, coastBack=True,
          f_weights = None
         ):
    
    best_v_result = 0.0
    backbones = nn.ModuleList([backbone_1, backbone_2])
    
    model = HTCNN_M(label_file, with_aux = True, with_fc = True, backbones=backbones,
              isCuda=True, isConditionProb=isConditionProb, coastBack=coastBack, weights=f_weights).to(device)
    
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name)
    
    output_filepath = os.path.join(output_path, output_fname)

    if checkpoint is not None and os.path.isfile(checkpoint):
        model_params = torch.load(checkpoint)
        model.load_state_dict(model_params)
        backbones = model.backbones
        print('Loaded from checkpoint %s'%checkpoint)
    
    #sample, _, _, _, _ = loadInBatch(trainset, batchsize = 1)
    #writer.add_graph(model, sample)
    #writer.close()
    
    v_result = 0
    
    
    model.eval()
    with torch.no_grad():
        val_result, aux_val_result, v_loss = computeAccuracy(valset, model, val_batch, withAux=True, preprocessor = v_preprocessor)
        v_result = val_result[0]
        print('Validation Accuracy: %f'%v_result)
        print(aux_val_result)
        best_v_result = v_result
    
    
    lr = start_lr
    optimizer = optim.SGD(model.parameters(), lr=lr)
    #optimizer = optim.Adagrad(model.parameters(), lr=lr)
    
    
    # create losses
    losses = nn.ModuleList()
    aux_loss_names = []
    aux_val_names = []
    final_loss = nn.MultiLabelSoftMarginLoss()
    for lv in lookup_lv_list:
        losses.append(nn.MultiLabelSoftMarginLoss())
        aux_loss_names.append('Coarst %d loss'%lv)
        aux_val_names.append('Level %d accuracy'%lv)
    losses.append(nn.MultiLabelSoftMarginLoss())
    aux_loss_names.append('Fine loss')
    aux_val_names.append('Fine accuracy')
    n_aux = len(losses) - 1
    aux_accuracy = {}
    
    for i in range(epoch):
        # training phase
        model.train()
        ptr = 0
        hasFinishEpoch = False
        epoch_result = []
        epoch_aux_losses_v = []
        epoch_loss_v = 0
        iter_c = 0
        avg_model_fwd_elapsed_time = 0.0
        while not hasFinishEpoch:
            optimizer.zero_grad()
            
            pp_start_time = time.time()
            batch_input, gt_aux, gt_final, ptr, hasFinishEpoch = loadInBatch(trainset, ptr, train_batch, shuffle=True,
                                                                            preprocessor=preprocessor, im_size = input_sizes)
            pp_elapsed_time = time.time() - pp_start_time
            
            model_start_time = time.time()
            pred_final, pred_aux = model(batch_input)
            model_fwd_elapsed_time = time.time() - model_start_time
            avg_model_fwd_elapsed_time = (avg_model_fwd_elapsed_time + model_fwd_elapsed_time) / 2.0
            
            iloss = 0
            fine_loss = losses[-1](pred_aux[-1], gt_final)*fine_scaler
            total_loss = fine_loss
            for i_aux in range(n_aux):
                aux_loss = losses[i_aux](pred_aux[i_aux], gt_aux[i_aux])
                total_loss = total_loss + aux_loss * aux_scaler
                aux_loss_v = aux_loss.item()
                if epoch_aux_losses_v == []:
                    epoch_aux_losses_v.append(aux_loss_v)
                else:
                    epoch_aux_losses_v[iloss] += aux_loss_v
                iloss += 1
            f_loss = final_loss(pred_final, gt_final)
            total_loss = total_loss + f_loss * final_scaler
            fine_loss_v = fine_loss.item()
            if len(epoch_aux_losses_v) <= iloss:
                epoch_aux_losses_v.append(fine_loss_v)
            else:
                epoch_aux_losses_v[iloss] += fine_loss_v
            # compute gradients
            total_loss.backward()
            
            # update weights
            optimizer.step()
            
            if iter_c == 0:
                epoch_loss_v = total_loss.item()
            else:
                epoch_loss_v += total_loss.item()
            
            if epoch_loss_v == 0:
                epoch_loss_v = total_loss.item()
            
            result = computeBatchAccuracy([pred_final],[gt_final])
            if epoch_result == []:
                epoch_result = result
            else:
                epoch_result = accumulateList(epoch_result, result)
            iter_c += 1
            print('[iteration %d]Data Loading Time:%f seconds; Computation Time:%f seconds'%(iter_c,pp_elapsed_time, model_fwd_elapsed_time))
        
        #print('Training Loss:', end='')
        plot_loss = {}
        for iloss in range(n_aux+1):
            epoch_aux_losses_v[iloss] /= iter_c
            plot_loss[aux_loss_names[iloss]] = epoch_aux_losses_v[iloss]
            plotter.plot('loss', 'aux %d'%iloss,'Losses', i, epoch_aux_losses_v[iloss])
            #print('%s: %f, '%(aux_loss_names[iloss], epoch_aux_losses_v[iloss]), end='')
        epoch_loss_v /= iter_c
        #lot_loss['total loss'] = epoch_loss_v
        plotter.plot('loss', 'total','Total Loss', i, epoch_loss_v)
        #writer.add_scalars('training loss', 
        #                  plot_loss,
        #                  i)
        #print('Fine loss: %f'%epoch_loss_v)
        print(plot_loss)
        
        # validation phase
        if i % val_at == 0:
            print('Validating...')
            model.eval()
            with torch.no_grad():
                val_result, aux_val_result, v_loss = computeAccuracy(valset, model, val_batch, withAux=True, preprocessor = v_preprocessor)
                for iacc in range(len(aux_val_names)):
                    aux_accuracy[aux_val_names[iacc]] = aux_val_result[iacc]
                v_result = val_result[0]
                print('Validation Accuracy: %f'%v_result)
                print(aux_accuracy)
                if v_result > best_v_result:
                    print('Best model found and saving it.')
                    torch.save(model.state_dict(), output_filepath)
                    best_v_result = v_result
        if i in lr_steps:
            olr = lr
            lr *= lr_discount
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('learning rate has been discounted from %f to %f'%(olr, lr))
        for i_aux in range(len(aux_accuracy)):
            plotter.plot('acc','aux %d'%(i_aux),'Accuracy', i, aux_accuracy[aux_val_names[i_aux]])
        plotter.plot('acc','final','Final Accuracy', i, v_result)
        #writer.add_scalars('Auxiliary Accuracy', 
        #                  aux_accuracy,
        #                  i)
        #writer.add_scalar('Final Accuracy', 
        #                  v_result,
        #                  i)
            
    print('Model has been trained.')
    model = None



def main():
    
    is_cond = args.cond == 1
    aux_weight = args.aux_weight
    final_weight = args.final_weight
    fine_weight = args.fine_weight
    step_down = [int(sd) for sd in args.step_down.split(',')]
    nepoch = args.epoch
    val_at = args.val_at
    train_batch = args.train_batch
    val_batch = args.val_batch
    lr = args.lr
    coastBack = args.enable_coast == 1
    fusion_weights = None
    if args.fusion_weights is not None:
        fusion_weights = [float(val) for val in args.fusion_weights.split(',')]
    
    
    checkpoint_path = os.path.join(model_path, model_fname)
    
    train_set = loadData(ds_root_path, training_file)
    val_set = loadData(ds_root_path, val_file)
    print('Training set has been buffered.')
    
    if backbone is not None:
        train(train_set, val_set, label_filepath,
              output_path = model_path, output_fname = model_fname, 
              epoch=nepoch, val_at=val_at, lr_steps=step_down, aux_scaler=aux_weight, final_scaler=final_weight,
             train_batch=train_batch, val_batch=val_batch, checkpoint=checkpoint_path,
              start_lr=lr, coastBack=coastBack,
             preprocessor=preprocess, v_preprocessor=preprocess_v, isConditionProb = is_cond, f_weights = fusion_weights)
    else:
        if isinstance(preprocess, list):
            train_mb_in(train_set, val_set, label_filepath,
                  output_path = model_path, output_fname = model_fname, 
                  epoch=nepoch, val_at=val_at, lr_steps=step_down, aux_scaler=aux_weight, final_scaler=final_weight,
                 train_batch=train_batch, val_batch=val_batch, checkpoint=checkpoint_path,
                     start_lr=lr, coastBack=coastBack,
                 preprocessor=preprocess, v_preprocessor=preprocess_v,
                       im_size=input_sizes,
                       general_process=gp, isConditionProb = is_cond, f_weights = fusion_weights)
        else:
            train_mb(train_set, val_set, label_filepath,
                  output_path = model_path, output_fname = model_fname, 
                  epoch=nepoch, val_at=val_at, lr_steps=step_down, aux_scaler=aux_weight, final_scaler=final_weight,
                 train_batch=train_batch, val_batch=val_batch, checkpoint=checkpoint_path,
                     start_lr=lr, coastBack=coastBack,
                 preprocessor=preprocess, v_preprocessor=preprocess_v, isConditionProb = is_cond, f_weights = fusion_weights)
    
    backbone_1 = None
    backbone_2 = None
    torch.cuda.empty_cache()
    #writer.close()
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for HTNN')
    parser.add_argument('--cond', dest='cond', type=int, default=1,
                       help='identify it is a condition prob or not')
    parser.add_argument('--aux-weight', dest='aux_weight', type=float, default=1.0,
                       help='loss weight for auxiliry losses')
    parser.add_argument('--fine-weight', dest='fine_weight', type=float, default=1.0,
                       help='loss weight for fine loss')
    parser.add_argument('--final-weight', dest='final_weight', type=float, default=1.0,
                       help='loss weight for fused loss')
    parser.add_argument('--backbone', dest='backbone_net', default='alexnet',
                       help='define the backbone network(s)')
    parser.add_argument('--multi-input', dest='mult_in', type=int, default=0,
                       help='identify it is multiple input for multiple backbone networks')
    parser.add_argument('--data-root', dest='data_root', default='/datasets/vision/cifar100_clean',
                       help='define the root path for dataset')
    parser.add_argument('--train', dest='train_file', default='/datasets/vision/cifar100_clean/train.txt',
                       help='define the training filepath')
    parser.add_argument('--val', dest='val_file', default='/datasets/vision/cifar100_clean/val.txt',
                       help='define the validation filepath')
    parser.add_argument('--test', dest='test_file', default='/datasets/vision/cifar100_clean/val.txt',
                       help='define the testing filepath')
    parser.add_argument('--tree', dest='tree_file', default='/datasets/vision/cifar100_clean/tree.txt',
                       help='define the tree filepath')
    parser.add_argument('--dst', dest='dst', default='/models/cifar100_htcnn_alexnet_2',
                       help='define the output path')
    parser.add_argument('--checkpoint', dest='checkpoint', default=None,
                       help='define the checkpoint path')
    parser.add_argument('--epoch', dest='epoch', type=int, default=300,
                       help='define the number of maximum epoch')
    parser.add_argument('--step-down', dest='step_down', default=[100, 200],
                       help='define the step down epoch')
    parser.add_argument('--val-at', dest='val_at', type=int, default=5,
                       help='define the number of epoch for validation')
    parser.add_argument('--train-batch', dest='train_batch', type=int, default=2048,
                       help='define the batch size for training')
    parser.add_argument('--val-batch', dest='val_batch', type=int, default=1024,
                       help='define the batch size of validation')
    parser.add_argument('--lr', dest='lr', type=float, default=0.1,
                       help='define the starting learning rate')
    parser.add_argument('--enable-coast', dest='enable_coast', type=int, default=1,
                       help='enable the backpropagation of coast projection layers')
    parser.add_argument('--learn-coast', dest='learn_coast', type=int, default=1,
                       help='indentify the network learns the coast or not')
    parser.add_argument('--n-coast', dest='n_coast', type=int, default=0,
                       help='indentify the number of coast to learn')
    parser.add_argument('--fine-lr', dest='f_lr', type=float, default=0.1,
                       help='define the learning rate for the fine branch')
    parser.add_argument('--fine-lr-discount', dest='f_lr_discount', type=float, default=0.1,
                       help='define the learning rate discount for the fine branch')
    parser.add_argument('--fine-step-down', dest='f_step_down', default=[100, 200],
                       help='define the step down epoch for the fine branch')
    parser.add_argument('--fine-same', dest='f_same', type=int, default=1,
                       help='identify the learning rate schedule of fine branch is the same as auxilaries')
    parser.add_argument('--fusion-weights', dest='fusion_weights', default = None,
                       help='define weights of bin output for fusion')
    

    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    #label_filepath = '/datasets/vision/cifar100_clean/tree.txt'
    #label_filepath = '/datasets/dummy/set1/tree.txt'
    label_filepath = args.tree_file
    classTree, n_coarst, coarst_dims = trees.build_itree(label_filepath)
    lookup_lv_list = [i+1 for i in range(n_coarst)]
    n_fine = len(list(classTree.keys()))
    print(coarst_dims)
    
    #ds_root_path = '/datasets/vision/cifar100_clean'
    #training_file = '/datasets/vision/cifar100_clean/train.txt'
    #val_file = '/datasets/vision/cifar100_clean/val.txt'
    #test_file = '/datasets/vision/cifar100_clean/val.txt'
    ds_root_path = args.data_root
    training_file = args.train_file
    val_file = args.val_file
    test_file = args.test_file
    
    #model_path = '/models/cifar100_htcnn_alexnet_2'
    model_path = args.dst
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    model_fname = 'model.pth'
    
    #backbone_1 = LeNet5(n_classes=coarst_dims[0])
    #backbone_2 = LeNet5(n_classes=n_fine).cuda()
    #backbone_1 = AlexNet32(n_classes=coarst_dims[0])
    #backbone_2 = AlexNet32(n_classes=n_fine)
    
    #backbone_2 = AlexNet32_B(n_classes=n_fine)
    backbone_2 = AlexNet32_C(n_classes=n_fine, feature_dim = 384)
    
    #backbone_1 = AlexNet32_D(n_classes=coarst_dims[0], fdim=64)
    #backbone_2 = AlexNet32_D(n_classes=n_fine, feature_dim=384)
    #backbone_1.common_features = backbone_2.common_features
    backbone = backbone_2
    #backbone = None
    backbone_inshape = backbone_2.input_dim
    
    gp = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5)
    ])
    
    preprocess_1 = transforms.Compose([
        transforms.Resize(40),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    preprocess_2 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    preprocess_3 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ])
    
    preprocess_4 = Compose([
        PadIfNeeded(40, 40, cv2.BORDER_CONSTANT, 0),
        RandomCrop(32,32),
        HorizontalFlip(p=0.5),
        ToTensor(normalize={
            'std':[0.2023, 0.1994, 0.2010],
            'mean':[0.4914, 0.4822, 0.4465]
        })
    ])
    preprocess_5 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    ])
    
    preprocess_v = Compose([
        ToTensor(normalize={
            'std':[0.2023, 0.1994, 0.2010],
            'mean':[0.4914, 0.4822, 0.4465]
        })
    ])
    
    #preprocess = [preprocess_1, preprocess_2]
    preprocess = preprocess_4
    input_sizes = [(3,32,32), (3,224,224)]
    input_sizes = input_sizes[0]
    #writer = SummaryWriter(log_dir = '../training', purge_step = 0,
    #                      flush_secs = 5)
    
    global plotter
    plotter = utils.VisdomLinePlotter(env_name='Hierarchy Tree Neural Network')
    #main()
    try:
        #pass
        main()
    except Exception as e:
        backbone_1 = None
        backbone_2 = None
        model = None
        torch.cuda.empty_cache()
        print('Error:')
        print(str(e))
        exit(-1)
