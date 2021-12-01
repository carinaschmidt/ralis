import os
import gc
import numpy as np
import pickle

import torch
import torchvision.transforms as standard_transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn

import utils.transforms as extended_transforms
import utils.joint_transforms_acdc as joint_transforms_acdc
import utils.transforms_acdc as extended_transforms_acdc
from data import brats18_2D, cityscapes, camvid, acdc, msdHeart, brats18, utils_acdc
from utils.logger import Logger
from utils.progressbar import progress_bar

import models.bratsUtils as bratsUtils
from models.bratsUtils import bratsDiceLoss, bratsDiceLossOriginal5, dice, sensitivity, specificity

from sklearn.metrics import confusion_matrix

import utils.parser as parser


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        print("dir_name: ", dir_name)
        os.mkdir(dir_name)


def create_and_load_optimizers(net, opt_choice, lr, wd,
                               momentum, ckpt_path, exp_name_toload, exp_name,
                               snapshot, checkpointer, load_opt,
                               policy_net=None, lr_dqn=0.0001, al_algorithm='random'):
    optimizerP = None
    opt_kwargs = {"lr": lr,
                  "weight_decay": wd,
                  "momentum": momentum
                  }
    opt_kwargs_rl = {"lr": lr_dqn,
                     "weight_decay": 0.001,
                     "momentum": momentum
                     }

    optimizer = optim.SGD(
        params=filter(lambda p: p.requires_grad, net.parameters()),
        **opt_kwargs)
    #import ipdb 
    #ipdb.set_trace()
    if al_algorithm == 'ralis' and policy_net is not None:
        if opt_choice == 'SGD':
            optimizerP = optim.SGD(
                params=filter(lambda p: p.requires_grad, policy_net.parameters()),
                **opt_kwargs_rl)
        elif opt_choice == 'RMSprop':
            optimizerP = optim.RMSprop(
                params=filter(lambda p: p.requires_grad, policy_net.parameters()),
                lr=lr_dqn)
        elif opt_choice == 'Adam':
            optimizerP = optim.Adam(
                params=filter(lambda p: p.requires_grad, policy_net.parameters()),
                lr=lr_dqn)
        elif opt_choice == 'AdamW':
            optimizerP = optim.AdamW(
                params=filter(lambda p: p.requires_grad, policy_net.parameters()),
                lr=lr_dqn)

    name = exp_name_toload if load_opt and len(exp_name_toload) > 0 else exp_name
    opt_path = os.path.join(ckpt_path, name, 'opt_' + snapshot)
    opt_policy_path = os.path.join(ckpt_path, name, 'opt_policy_' + snapshot)

    if (load_opt and len(exp_name_toload)) > 0 or (checkpointer and os.path.isfile(opt_path)):
        print('(Opt load) Loading net optimizer')
        optimizer.load_state_dict(torch.load(opt_path))

        if al_algorithm == 'ralis' and os.path.isfile(opt_policy_path) and optimizerP is not None: #@carina added and optimizerP is not None
            print('(Opt load) Loading policy optimizer')
            optimizerP.load_state_dict(torch.load(opt_policy_path))

    print ('Optimizers created')
    return optimizer, optimizerP


def get_logfile(ckpt_path, exp_name, checkpointer, snapshot, num_classes=19, log_name='log.txt'):
    args = parser.get_arguments()
    if args.modality == '2D':
        if 'brats18' or 'acdc' in args.dataset:
            log_columns = ['Epoch', 'Learning Rate', 'Train Loss', '(deprecated)',
                        'Valid Loss', 'Train Acc.', 'Valid Acc.',
                        'Train mean dice', 'Valid mean dice']
            for cl in range(num_classes):
                log_columns.append('dice_cl' + str(cl))

            best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'mean_dice': 0}
            curr_epoch = 0
            ##-- Check if log file exists --##
            if checkpointer:
                if os.path.isfile(os.path.join(ckpt_path, exp_name, log_name)):
                    print('(Checkpointer) Log file ' + log_name + ' already exists, appending.')
                    logger = Logger(os.path.join(ckpt_path, exp_name, log_name),
                                    title=exp_name, resume=True)
                    if 'best' in snapshot:
                        curr_epoch = int(logger.resume_epoch)
                    else:
                        curr_epoch = logger.last_epoch
                    best_record = {'epoch': int(logger.resume_epoch), 'val_loss': 1e10,
                                'mean_dice': float(logger.resume_jacc), 'acc': 0}
                else:
                    print('(Checkpointer) Log file ' + log_name + ' did not exist before, creating')
                    logger = Logger(os.path.join(ckpt_path, exp_name, log_name),
                                    title=exp_name)
                    logger.set_names(log_columns)

            else:
                print('(No checkpointer activated) Log file ' + log_name + ' created.')
                logger = Logger(os.path.join(ckpt_path, exp_name, log_name),
                                title=exp_name)
                logger.set_names(log_columns)
        else: # for not brats18 2D data 
            log_columns = ['Epoch', 'Learning Rate', 'Train Loss', '(deprecated)',
                        'Valid Loss', 'Train Acc.', 'Valid Acc.',
                        'Train mean iu', 'Valid mean iu']
            for cl in range(num_classes):
                log_columns.append('iu_cl' + str(cl))

            best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'mean_iu': 0}
            curr_epoch = 0
            ##-- Check if log file exists --##
            if checkpointer:
                if os.path.isfile(os.path.join(ckpt_path, exp_name, log_name)):
                    print('(Checkpointer) Log file ' + log_name + ' already exists, appending.')
                    logger = Logger(os.path.join(ckpt_path, exp_name, log_name),
                                    title=exp_name, resume=True)
                    if 'best' in snapshot:
                        curr_epoch = int(logger.resume_epoch)
                    else:
                        curr_epoch = logger.last_epoch
                    best_record = {'epoch': int(logger.resume_epoch), 'val_loss': 1e10,
                                'mean_iu': float(logger.resume_jacc), 'acc': 0}
                else:
                    print('(Checkpointer) Log file ' + log_name + ' did not exist before, creating')
                    logger = Logger(os.path.join(ckpt_path, exp_name, log_name),
                                    title=exp_name)
                    logger.set_names(log_columns)

            else:
                print('(No checkpointer activated) Log file ' + log_name + ' created.')
                logger = Logger(os.path.join(ckpt_path, exp_name, log_name),
                                title=exp_name)
                logger.set_names(log_columns)
        return logger, best_record, curr_epoch
    


    else: # '3D' #
        log_columns = ['Epoch', 'Learning Rate', 'Train Loss', 'Train Acc(deprecated)', 'Train mean dice',
            'Valid Loss', 'Valid Acc', 'Valid mean dice', 'Valid dice WT', 'Valid dice TC', 'Valid dice ET']
        best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'mean_dice': 0}
        curr_epoch = 0

        ##-- Check if log file exists --##
        if checkpointer:
            if os.path.isfile(os.path.join(ckpt_path, exp_name, log_name)):
                print('(Checkpointer) Log file ' + log_name + ' already exists, appending.')
                logger = Logger(os.path.join(ckpt_path, exp_name, log_name),
                                title=exp_name, resume=True)
                if 'best' in snapshot:
                    curr_epoch = int(logger.resume_epoch)
                else:
                    curr_epoch = logger.last_epoch
                best_record = {'epoch': int(logger.resume_epoch), 'val_loss': 1e10,
                            'mean_dice': float(logger.resume_jacc), 'acc': 0}
            else:
                print('(Checkpointer) Log file ' + log_name + ' did not exist before, creating')
                logger = Logger(os.path.join(ckpt_path, exp_name, log_name),
                                title=exp_name)
                logger.set_names(log_columns)

        else:
            print('(No checkpointer activated) Log file ' + log_name + ' created.')
            logger = Logger(os.path.join(ckpt_path, exp_name, log_name),
                            title=exp_name)
            logger.set_names(log_columns)
        return logger, best_record, curr_epoch


def get_training_stage(args):
    path = os.path.join(args.ckpt_path, args.exp_name,
                        'training_stage.pkl')
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            stage = pickle.load(f)
    else:
        stage = None
    return stage


def set_training_stage(args, stage):
    path = os.path.join(args.ckpt_path, args.exp_name,
                        'training_stage.pkl')
    with open(path, 'wb') as f:
        pickle.dump(stage, f)
    
def evaluate(cm):
    args = parser.get_arguments()
    #import ipdb
    #ipdb.set_trace() 
    # Compute metrics 
    #@carina  diagonal elements show the number of correct classifications for each class, if 0: no correct classifications!
    #print("cm: ", cm)
    TP_perclass = cm.diagonal().astype('float32')
    #jaccard_perclass = TP_perclass / (cm.sum(1) + cm.sum(0) - TP_perclass)
    jaccard_perclass = np.divide(TP_perclass, (cm.sum(1) + cm.sum(0) - TP_perclass))  #overlap (true predictions, intersection) / union
    jaccard = np.mean(jaccard_perclass)
    accuracy = TP_perclass.sum() / cm.sum()  #diagonal: correct predicted / all predictions


    ### Dice score 2D
    print("args.dataset: ", args.dataset)
    print("args.modality", args.modality)
    if ('acdc' in args.dataset or 'brats18' in args.dataset) and '2D' in args.modality:
        print("jaccard_per_class: ", jaccard_perclass)
        dice_perclass = np.divide((2*jaccard_perclass), (jaccard_perclass + 1))
        print("dice per class: ", dice_perclass)
        dice = np.mean(dice_perclass)
        return accuracy, dice, dice_perclass
    
    else:
        print("evaluating jaccard not dice !!!")
        return accuracy, jaccard, jaccard_perclass


def confusion_matrix_pytorch(cm, output_flatten, target_flatten, num_classes):
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = cm[i, j] + ((output_flatten == i) * (target_flatten == j)).sum().type(torch.IntTensor).cuda()
    return cm



def compute_set_jacc(val_loader, net):
    """Compute accuracy, mean IoU and IoU per class on the provided set.
    :param dataset_target: Dataset
    :param net: Classification network
    :return: accuracy (float), iou (float), iou per class (list of floats)
    """
    net.eval()

    cm_py = torch.zeros((val_loader.dataset.num_classes, val_loader.dataset.num_classes)).type(
        torch.IntTensor).cuda()
    for vi, data in enumerate(val_loader):
        inputs, gts_, _ = data
        with torch.no_grad():
            inputs = Variable(inputs).cuda()

        outputs, _ = net(inputs)

        if outputs.shape[2:] != gts_.shape[1:]:
            outputs = outputs[:, :, 0:min(outputs.shape[2], gts_.shape[1]), 0:min(outputs.shape[3], gts_.shape[2])]
            gts_ = gts_[:, 0:min(outputs.shape[2], gts_.shape[1]), 0:min(outputs.shape[3], gts_.shape[2])]
        predictions_py = outputs.data.max(1)[1].squeeze_(1)
        #import ipdb 
        #ipdb.set_trace()
        cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),        #contains TP, TN, FP, FN.. from target and predicted values
                                         gts_.cuda().view(-1),
                                         val_loader.dataset.num_classes)

        del (outputs)
        del (predictions_py)
    acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())

    return acc, mean_iu, iu


def train(train_loader, net, criterion, optimizer, supervised=False):
    torch.cuda.empty_cache()
    args = parser.get_arguments()
    if args.modality == '2D': #and 'acdc' in args.dataset:
        net.train()
        train_loss = 0
        cm_py = torch.zeros((train_loader.dataset.num_classes, train_loader.dataset.num_classes)).type(
            torch.IntTensor).cuda()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if supervised:
                im_s, t_s_, _ = data    # im_s.shape = [32, 3, 128, 128], t_s_.shape = [32, 128, 128]
                #print("im_s.shape: ", im_s.shape) #brats im_s.shape:  torch.Size([2, 4, 128, 128, 128])
                #print("t_s_.shape: ", t_s_.shape) #brats t_s_.shape:  torch.Size([2, 3, 128, 128, 128]) or [2,1] or  [1,3]...
            else:
                im_s, t_s_, _, _, _ = data   #im_s.shape: torch.Size([16, 256, 256, 3]), t_s_: torch.Size([16, 256, 256]), #t_s_.unique(): tensor([0, 1, 2, 3, 4])

            #print("t_s_: ", np.unique(t_s_))
            if t_s_.max() >= torch.tensor(4) and 'acdc' in args.dataset:
                print("ATTENTION! max value is 4, need to change!")
                t_s_[t_s_==4] = 3
                #print("new t_s.max(): ", t_s_.max())   
            if t_s_.min() < torch.tensor(0) and ('acdc' or 'brats18') in args.dataset:
                print("ATTENTION! min value is negative!: ", t_s_.min())
                t_s_[t_s_==-1] = 0
            
            t_s, im_s = Variable(t_s_).cuda(), Variable(im_s).cuda() 
            # Get output of network
            outputs, _ = net(im_s) 
            # Get segmentation maps
            # print("outputs.shape: ", outputs.shape) # torch.Size([2, 4, 152, 152])|
            predictions_py = outputs.data.max(1)[1].squeeze_(1) #brats18: [2, 152, 152]      ACDC: [16, 128, 128]
            #predictions_py_stacked = torch.stack((predictions_py,predictions_py,predictions_py), dim=1)
            #print("pred_stacked: ", predictions_py_stacked)
            #print("outputs: ", outputs.shape) #([2, 4, 128, 128])
            #print("t_s.shape: ", t_s.shape) # torch.Size([2, 3, 128, 128])
            #t_s = torch.squeeze(t_s)
            #print("t_s.shape: ", t_s.shape)
            #loss = bratsDiceLossOriginal5(outputs, t_s) 
            #loss = bratsDiceLoss(outputs, t_s) 
            #print("predictions_py: ", predictions_py)
            loss = criterion(outputs, t_s)

            train_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=4)
            optimizer.step()

            cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1), 
                                            t_s_.cuda().view(-1),
                                            train_loader.dataset.num_classes)

            progress_bar(i, len(train_loader), '[train loss %.5f]' % (
                    train_loss / (i + 1)))

            del (outputs)
            del (loss)
            gc.collect()
            torch.cuda.empty_cache()
        print(' ')
        #import ipdb
        #ipdb.set_trace()
        acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())
        if 'brats18' or 'acdc' in args.dataset:
            print(' [train acc %.5f], [train dice %.5f]' % (
                acc, mean_iu))
        else: 
            print(' [train acc %.5f], [train iu %.5f]' % (
                acc, mean_iu))
        return train_loss / (len(train_loader)), 0, acc, mean_iu
    
    # for '3D': BraTS 2018
    else:
        print("in else")
        net.train()
        train_loss = 0
        cm_py = torch.zeros((train_loader.dataset.num_classes, train_loader.dataset.num_classes)).type(
            torch.IntTensor).cuda()

        diceWT, diceTC, diceET = [], [], []
        sensWT, sensTC, sensET = [], [], []
        specWT, specTC, specET = [], [], []
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            if supervised:
                im_s, t_s_, (img_path, mask_path, im_name) = data    # im_s.shape = [2,4,128,128,128], t_s_.shape = [2,1,128,128,128]
            else:
                im_s, t_s_, _, _, _ = data   #im_s.shape: torch.Size([16, 256, 256, 3]), t_s_: torch.Size([16, 256, 256]), #t_s_.unique(): tensor([0, 1, 2, 3, 4])

            #if outputs.shape != t_s_.shape:
            t_s_stacked = torch.cat((t_s_, t_s_, t_s_), dim=1) #stacks to get shape [2,3,128,128,128]
            
            t_s, im_s = Variable(t_s_stacked).cuda(), Variable(im_s).cuda() 
            # Get output of network)
            outputs = net(im_s) 

            # Get segmentation maps
            #print("outputs.shape: ", outputs.shape)  #BRATS: [2, 3, 128, 128, 128]     ACDC: [16, 4, 128, 128]
            predictions_py = outputs.data.max(1)[1].squeeze_(1) # torch.Size([2, 128, 128, 128]) [16, 128, 128]
            #print("predictions_py.shape: ", predictions_py.shape) # [2, 128,128,128]
            # print("max of pred_py: ", predictions_py.max())  #max 2
            # print("min of pred_py: ", predictions_py.min()) #min 0
            # print("t_s.shape", t_s.shape)       #[2,3,128,128,18]
            # print("outputs.shape: ", outputs.shape)  #BRATS: [2, 3, 128, 128, 128]     ACDC: [16, 4, 128, 128]

            # use bratsDiceloss 
            #loss = criterion(outputs, t_s)
            loss = bratsDiceLoss(outputs, t_s) 

            train_loss += loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=4)
            optimizer.step()

            # predictions_py and t_s_ must match the size
            cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
                                            t_s_.cuda().view(-1),
                                            train_loader.dataset.num_classes)

            progress_bar(i, len(train_loader), '[train loss %.5f]' % (
                    train_loss / (i + 1)))


            #separate outputs channelwise
            wt, tc, et = outputs.chunk(3, dim=1)
            s = wt.shape
            wt = wt.view(s[0], s[2], s[3], s[4])
            tc = tc.view(s[0], s[2], s[3], s[4])
            et = et.view(s[0], s[2], s[3], s[4])

            wtMask, tcMask, etMask = outputs.chunk(3, dim=1)
            s = wtMask.shape
            wtMask = wtMask.view(s[0], s[2], s[3], s[4])
            tcMask = tcMask.view(s[0], s[2], s[3], s[4])
            etMask = etMask.view(s[0], s[2], s[3], s[4])

            #get dice metrics
            diceWT.append(bratsUtils.dice(wt, wtMask))
            diceTC.append(bratsUtils.dice(tc, tcMask))
            diceET.append(bratsUtils.dice(et, etMask))

            #get sensitivity metrics
            sensWT.append(bratsUtils.sensitivity(wt, wtMask))
            sensTC.append(bratsUtils.sensitivity(tc, tcMask))
            sensET.append(bratsUtils.sensitivity(et, etMask))

            #get specificity metrics
            specWT.append(bratsUtils.specificity(wt, wtMask))
            specTC.append(bratsUtils.specificity(tc, tcMask))
            specET.append(bratsUtils.specificity(et, etMask))

            # print("in train: diceWT: ", diceWT)

            del (outputs)
            del (loss)

            del (wt)
            del (wtMask)
            del (tc)
            del (tcMask)
            del (et)
            del (etMask)

            gc.collect()
            torch.cuda.empty_cache()
        print(' ')
        #import ipdb
        #ipdb.set_trace()

        #calculate mean dice scores
        meanDiceWT = np.mean(diceWT)
        meanDiceTC = np.mean(diceTC)
        meanDiceET = np.mean(diceET)
        meanDice = np.mean([meanDiceWT, meanDiceTC, meanDiceET])

        # print("in train: meanDiceWT: ", meanDiceWT)

        del (diceWT)
        del (diceTC)
        del (diceET)

        # calculate mean sensitivity and specificity
        meanSensWT = np.mean(sensWT)
        meanSensTC = np.mean(sensTC)
        meanSensET = np.mean(sensET)
        meanSensWT = np.mean([meanSensWT, meanSensTC, meanSensET])

        del (sensWT)
        del (sensTC)
        del (sensET)

        acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())
        
        print(' [train acc %.5f], [train iu %.5f], [mean dice %.5f]' % (
            acc, mean_iu, meanDice))


        return train_loss / (len(train_loader)), 0, acc, mean_iu, meanDice


def validate(val_loader, net, criterion, optimizer, epoch, best_record, args, final_final_test=False):
    torch.cuda.empty_cache()
    net.eval()
    ##### 2D #####
    if args.modality == '2D':           
        val_loss = 0
        cm_py = torch.zeros((val_loader.dataset.num_classes, val_loader.dataset.num_classes)).type(
            torch.IntTensor).cuda()
        # import ipdb
        # ipdb.set_trace() 
        for vi, data in enumerate(val_loader):
            inputs, gts_, _ = data #inputs.shape = torch.Size([1, 3, 256, 256]), gts_.shape: torch.Size([1, 256, 256])

            with torch.no_grad():
                inputs = Variable(inputs).cuda()
                gts = Variable(gts_).cuda()
            
            #print("inputs before net: ", inputs.shape)
            outputs, _ = net(inputs) #outputs.shape: torch.Size([1, 4, 256, 256])

            # Make sure both output and target have the same dimensions
            if outputs.shape[2:] != gts.shape[1:]:   ##gts.shape torch.Size([1, 256, 256])
                outputs = outputs[:, :, 0:min(outputs.shape[2], gts.shape[1]), 0:min(outputs.shape[3], gts.shape[2])] 
                gts = gts[:, 0:min(outputs.shape[2], gts.shape[1]), 0:min(outputs.shape[3], gts.shape[2])] 
            predictions_py = outputs.data.max(1)[1].squeeze_(1)  #ipdb> predictions_py.shape: torch.Size([1, 256, 256])
            #print("outputs.shape: ", outputs.shape)
            #print("gts.shape: ", gts.shape)
            loss = criterion(outputs, gts)
            vl_loss = loss.item()
            val_loss += (vl_loss)

            cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
                                            gts_.cuda().view(-1),
                                            val_loader.dataset.num_classes)

            len_val = len(val_loader)
            progress_bar(vi, len_val, '[val loss %.5f]' % (
                    val_loss / (vi + 1)))
            del (outputs)
            del (vl_loss)
            del (loss)
            del (predictions_py)
        acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())
        if 'brats18' or 'acdc' in args.dataset:
            print('  ')
            print(' [val acc %.5f], [val dice %.5f]' % (
            acc, mean_iu))

        else:
            print(' ')
            print(' [val acc %.5f], [val iu %.5f]' % (
                acc, mean_iu))

        if not final_final_test:
            if 'brats18' or 'acdc' in args.dataset:
                if mean_iu > best_record['mean_dice']:
                    best_record['val_loss'] = val_loss / len(val_loader)
                    best_record['epoch'] = epoch
                    best_record['acc'] = acc
                    best_record['dice'] = iu
                    best_record['mean_dice'] = mean_iu

                    torch.save(net.cpu().state_dict(),
                            os.path.join(args.ckpt_path, args.exp_name,
                                            'best_jaccard_val.pth'))
                    net.cuda()
                    torch.save(optimizer.state_dict(),
                            os.path.join(args.ckpt_path, args.exp_name,
                                            'opt_best_jaccard_val.pth'))

                ## Save checkpoint every epoch
                #@carina will be overwritten
                torch.save(net.cpu().state_dict(),
                        os.path.join(args.ckpt_path, args.exp_name,
                                        'last_jaccard_val.pth'))

                # save model every 50 epochs
                if epoch % 50 == 0:
                    torch.save(net.cpu().state_dict(),
                        os.path.join(args.ckpt_path, args.exp_name,
                                    'jaccard_val_epoch{}.pth'.format(epoch)))
                net.cuda()
                torch.save(optimizer.state_dict(),
                        os.path.join(args.ckpt_path, args.exp_name,
                                        'opt_last_jaccard_val.pth'))

                print(
                        'best record: [val loss %.5f], [acc %.5f], [mean_dice %.5f],'
                        ' [epoch %d]' % (best_record['val_loss'], best_record['acc'],
                                        best_record['mean_dice'], best_record['epoch']))
            else: #not brats18 or acdc in dataset
                if mean_iu > best_record['mean_iu']:
                    best_record['val_loss'] = val_loss / len(val_loader)
                    best_record['epoch'] = epoch
                    best_record['acc'] = acc
                    best_record['iu'] = iu
                    best_record['mean_iu'] = mean_iu

                    torch.save(net.cpu().state_dict(),
                            os.path.join(args.ckpt_path, args.exp_name,
                                            'best_jaccard_val.pth'))
                    net.cuda()
                    torch.save(optimizer.state_dict(),
                            os.path.join(args.ckpt_path, args.exp_name,
                                            'opt_best_jaccard_val.pth'))

                ## Save checkpoint every epoch
                #@carina will be overwritten
                torch.save(net.cpu().state_dict(),
                        os.path.join(args.ckpt_path, args.exp_name,
                                        'last_jaccard_val.pth'))

                # save model every 50 epochs
                if epoch % 50 == 0:
                    torch.save(net.cpu().state_dict(),
                        os.path.join(args.ckpt_path, args.exp_name,
                                    'jaccard_val_epoch{}.pth'.format(epoch)))
                net.cuda()
                torch.save(optimizer.state_dict(),
                        os.path.join(args.ckpt_path, args.exp_name,
                                        'opt_last_jaccard_val.pth'))

                print(
                        'best record: [val loss %.5f], [acc %.5f], [mean_iu %.5f],'
                        ' [epoch %d]' % (best_record['val_loss'], best_record['acc'],
                                        best_record['mean_iu'], best_record['epoch']))

        print('----------------------------------------')
        return val_loss / len(val_loader), acc, mean_iu, iu, best_record

    # 3D modality
    else:
        print("validation for 3d")
        val_loss = 0
        cm_py = torch.zeros((val_loader.dataset.num_classes, val_loader.dataset.num_classes)).type(
            torch.IntTensor).cuda()

        
        diceWT, diceTC, diceET = [], [], []
        sensWT, sensTC, sensET = [], [], []
        specWT, specTC, specET = [], [], []

        for vi, data in enumerate(val_loader):
            inputs, gts_, _ = data #inputs.shape = torch.Size([1, 3, 256, 256]), gts_.shape: torch.Size([1, 256, 256])

            with torch.no_grad():
                inputs = Variable(inputs).cuda()
                #gts = Variable(gts_).cuda()
            outputs = net(inputs) #outputs.shape: torch.Size([1, 4, 256, 256])   
            #print("outputs.shape: ", outputs.shape) #[2, 3, 128, 128, 128] (for batch size = 2)
            #print("gts_.shape: ", gts_.shape) # [1, 1, 128, 128, 128]
            # Make sure both output and target have the same dimensions
            if outputs.shape != gts_.shape:   
                #gts_3ch = gts_.squeeze() #remove dim of size 1
                gts_3ch = gts_
                gts_3ch = torch.cat((gts_3ch, gts_3ch, gts_3ch), dim=1) #stacks to get shape [1, 3, 128, 128, 128]
                gts = Variable(gts_3ch).cuda()

            predictions_py = outputs.data.max(1)[1].squeeze_(1)  #ipdb> predictions_py.shape: torch.Size([1, 256, 256])
            loss = bratsDiceLoss(outputs, gts) 
            #print("loss: ", loss.item())
            vl_loss = loss.item()
            val_loss += (vl_loss)

            cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
                                            gts_.cuda().view(-1),
                                            val_loader.dataset.num_classes)



            len_val = len(val_loader)
            progress_bar(vi, len_val, '[val loss %.5f]' % (
                    val_loss / (vi + 1)))


            #separate outputs channelwise
            print("outputs.shape: ", outputs.shape)
            print("wt.shape: ", wt.shape)
            print("tc.shape: ", tc.shape)
            wt, tc, et = outputs.chunk(3, dim=1)
            s = wt.shape
            wt = wt.view(s[0], s[2], s[3], s[4])
            tc = tc.view(s[0], s[2], s[3], s[4])
            et = et.view(s[0], s[2], s[3], s[4])

            wtMask, tcMask, etMask = outputs.chunk(3, dim=1)
            s = wtMask.shape
            wtMask = wtMask.view(s[0], s[2], s[3], s[4])
            tcMask = tcMask.view(s[0], s[2], s[3], s[4])
            etMask = etMask.view(s[0], s[2], s[3], s[4])

            #get dice metrics
            diceWT.append(bratsUtils.dice(wt, wtMask))
            diceTC.append(bratsUtils.dice(tc, tcMask))
            diceET.append(bratsUtils.dice(et, etMask))

            #get sensitivity metrics
            sensWT.append(bratsUtils.sensitivity(wt, wtMask))
            sensTC.append(bratsUtils.sensitivity(tc, tcMask))
            sensET.append(bratsUtils.sensitivity(et, etMask))

            #get specificity metrics
            specWT.append(bratsUtils.specificity(wt, wtMask))
            specTC.append(bratsUtils.specificity(tc, tcMask))
            specET.append(bratsUtils.specificity(et, etMask))

            # print("in validate: diceWT: ", diceWT)

            del (outputs)
            del (vl_loss)
            del (loss)
            del (predictions_py)
            del (wt)
            del (tc)
            del (et)
            del (s)
            del (wtMask)
            del (tcMask)
            del (etMask)

        #calculate mean dice scores
        meanDiceWT = np.mean(diceWT)
        meanDiceTC = np.mean(diceTC)
        meanDiceET = np.mean(diceET)
        meanDice = np.mean([meanDiceWT, meanDiceTC, meanDiceET])

        # print("in validate: mean diceWT: ", meanDiceWT)

        del (diceWT)
        del (diceTC)
        del (diceET)


        acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())
        print(' ')
        print(' [val acc %.5f], [val iu %.5f], [val mean dice %.5f]' % (
            acc, mean_iu, meanDice))

        if not final_final_test:
            if meanDice > best_record['mean_dice']:
                best_record['val_loss'] = val_loss / len(val_loader)
                best_record['epoch'] = epoch
                best_record['acc'] = acc
                best_record['iu'] = iu
                best_record['mean_iu'] = mean_iu

                ### new for BraTS
                best_record['mean_dice'] = meanDice
                best_record['mean_dice_WT'] = meanDiceWT
                best_record['mean_dice_TC'] = meanDiceTC
                best_record['mean_dice_ET'] = meanDiceET

                torch.save(net.cpu().state_dict(),
                        os.path.join(args.ckpt_path, args.exp_name,
                                        'best_jaccard_val.pth'))
                net.cuda()
                torch.save(optimizer.state_dict(),
                        os.path.join(args.ckpt_path, args.exp_name,
                                        'opt_best_jaccard_val.pth'))

            ## Save checkpoint every epoch
            #@carina will be overwritten
            torch.save(net.cpu().state_dict(),
                    os.path.join(args.ckpt_path, args.exp_name,
                                    'last_jaccard_val.pth'))

            # save model every 50 epochs
            if epoch % 50 == 0:
                torch.save(net.cpu().state_dict(),
                    os.path.join(args.ckpt_path, args.exp_name,
                                'jaccard_val_epoch{}.pth'.format(epoch)))
            net.cuda()
            torch.save(optimizer.state_dict(),
                    os.path.join(args.ckpt_path, args.exp_name,
                                    'opt_last_jaccard_val.pth'))

            print(
                    'best record: [val loss %.5f], [acc %.5f], [mean_dice %.5f],'
                    ' [epoch %d]' % (best_record['val_loss'], best_record['acc'],
                                    best_record['mean_dice'], best_record['epoch']))

        print('----------------------------------------')

        return val_loss / len(val_loader), acc, mean_iu, iu, meanDice, meanDiceWT, meanDiceTC, meanDiceET, best_record


def test(val_loader, net, criterion):
    net.eval()
    args = parser.get_arguments()
    if args.modality == '2D':   
        val_loss = 0
        cm_py = torch.zeros((val_loader.dataset.num_classes, val_loader.dataset.num_classes)).type(
            torch.IntTensor).cuda()

        #print(val_loader)
        for vi, data in enumerate(val_loader):
            inputs, gts_, _ = data
            with torch.no_grad():
                # print("inputs: ", inputs.shape)
                # print("gts: ", gts_.shape)
                inputs = Variable(inputs).cuda()
                gts = Variable(gts_).cuda()

            outputs, _ = net(inputs)
            # @carina Make sure both output and target have the same dimensions
            if outputs.shape[2:] != gts.shape[1:]:   ##gts.shape torch.Size([1, 256, 256])
                print("outputs and gts did not have the same shape")
                outputs = outputs[:, :, 0:min(outputs.shape[2], gts.shape[1]), 0:min(outputs.shape[3], gts.shape[2])] 
                gts = gts[:, 0:min(outputs.shape[2], gts.shape[1]), 0:min(outputs.shape[3], gts.shape[2])] 

            predictions_py = outputs.data.max(1)[1].squeeze_(1)

            loss = criterion(outputs, gts)
            vl_loss = loss.item()
            val_loss += (vl_loss)

            cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
                                            gts_.cuda().view(-1),
                                            val_loader.dataset.num_classes)

            len_val = len(val_loader)
            progress_bar(vi, len_val, '[val loss %.5f]' % (
                    val_loss / (vi + 1)))

            del (outputs)
            del (vl_loss)
        acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())
        if 'brats18' or 'acdc' in args.dataset:
            print(' ')
            print(' [val acc %.5f], [val dice %.5f]' % (
                acc, mean_iu))
        else:
            print(' ')
            print(' [val acc %.5f], [val iu %.5f]' % (
                acc, mean_iu))

        return val_loss / len(val_loader), acc, mean_iu, iu

    ##### 3D ####
    else:
        val_loss = 0
        cm_py = torch.zeros((val_loader.dataset.num_classes, val_loader.dataset.num_classes)).type(
            torch.IntTensor).cuda()
        diceWT, diceTC, diceET = [], [], []
        sensWT, sensTC, sensET = [], [], []
        specWT, specTC, specET = [], [], []

        #print(val_loader)
        for vi, data in enumerate(val_loader):
            inputs, gts_, _ = data
            with torch.no_grad():
                # print("inputs: ", inputs.shape)
                # print("gts: ", gts_.shape)
                inputs = Variable(inputs).cuda()
                gts = Variable(gts_).cuda()

            outputs = net(inputs)
            # print("outputs.shape: ", outputs.shape) #[2, 3, 128, 128, 128] (for batch size = 2)
            # print("gts_.shape: ", gts_.shape) # [1, 1, 128, 128, 128]
            # Make sure both output and target have the same dimensions
            if outputs.shape != gts_.shape:   
                #gts_3ch = gts_.squeeze() #remove dim of size 1
                gts_3ch = gts_
                gts_3ch = torch.cat((gts_3ch, gts_3ch, gts_3ch), dim=1) #stacks to get shape [1, 3, 128, 128, 128]
                gts = Variable(gts_3ch).cuda()
            predictions_py = outputs.data.max(1)[1].squeeze_(1)

            loss = bratsDiceLoss(outputs, gts) 
            # print("loss: ", loss)

            vl_loss = loss.item()
            val_loss += (vl_loss)

            cm_py = confusion_matrix_pytorch(cm_py, predictions_py.view(-1),
                                            gts_.cuda().view(-1),
                                            val_loader.dataset.num_classes)

            len_val = len(val_loader)
            progress_bar(vi, len_val, '[val loss %.5f]' % (
                    val_loss / (vi + 1)))

            #separate outputs channelwise
            wt, tc, et = outputs.chunk(3, dim=1)
            s = wt.shape
            wt = wt.view(s[0], s[2], s[3], s[4])
            tc = tc.view(s[0], s[2], s[3], s[4])
            et = et.view(s[0], s[2], s[3], s[4])

            wtMask, tcMask, etMask = outputs.chunk(3, dim=1)
            s = wtMask.shape
            wtMask = wtMask.view(s[0], s[2], s[3], s[4])
            tcMask = tcMask.view(s[0], s[2], s[3], s[4])
            etMask = etMask.view(s[0], s[2], s[3], s[4])

            diceWT.append(bratsUtils.dice(wt, wtMask))
            diceTC.append(bratsUtils.dice(tc, tcMask))
            diceET.append(bratsUtils.dice(et, etMask))

            #get sensitivity metrics
            sensWT.append(bratsUtils.sensitivity(wt, wtMask))
            sensTC.append(bratsUtils.sensitivity(tc, tcMask))
            sensET.append(bratsUtils.sensitivity(et, etMask))

            #get specificity metrics
            specWT.append(bratsUtils.specificity(wt, wtMask))
            specTC.append(bratsUtils.specificity(tc, tcMask))
            specET.append(bratsUtils.specificity(et, etMask))

            del (outputs)
            del (vl_loss)
            del (wt)
            del (tc)
            del (et)
            del (s)
            del (wtMask)
            del (tcMask)
            del (etMask)

        #calculate mean dice scores
        meanDiceWT = np.mean(diceWT)
        meanDiceTC = np.mean(diceTC)
        meanDiceET = np.mean(diceET)
        meanDice = np.mean([meanDiceWT, meanDiceTC, meanDiceET])

        del (diceWT)
        del (diceTC)
        del (diceET)

        # calculate mean sensitivity and specificity
        meanSensWT = np.mean(sensWT)
        meanSensTC = np.mean(sensTC)
        meanSensET = np.mean(sensET)
        meanSens = np.mean([meanSensWT, meanSensTC, meanSensET])

        del (sensWT)
        del (sensTC)
        del (sensET)

        acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())
        print(' ')
        print(' [val acc %.5f], [val iu %.5f], [mean dice %.5f]' % (
            acc, mean_iu, meanDice))

        return val_loss / len(val_loader), acc, mean_iu, iu, meanDice, meanDiceWT, meanDiceTC, meanDiceET


def final_test(args, net, criterion):
    # Load best checkpoint for segmentation network
    net_checkpoint_path = os.path.join(args.ckpt_path, args.exp_name, 'best_jaccard_val.pth')
    if os.path.isfile(net_checkpoint_path):
        print('(Final test) Load best checkpoint for segmentation network!')
        net_dict = torch.load(net_checkpoint_path)
        if len([key for key, value in net_dict.items() if 'module' in key.lower()]) > 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in net_dict.items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            net_dict = new_state_dict
        net.load_state_dict(net_dict)
    net.eval()

    # Prepare data transforms
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if 'acdc' or 'msdHeart' or 'brats18' in args.dataset:
        #@carina added
        input_transform = None
        target_transform = None
        # input_transform = standard_transforms.Compose([
        #     extended_transforms_acdc.ImageToTensor()
        # ])
        # target_transform = extended_transforms_acdc.MaskToTensor()
    else:
        input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        target_transform = extended_transforms.MaskToTensor()

    if 'camvid' in args.dataset:
        val_set = camvid.Camvid('fine', 'test' if test else 'val',
                                data_path=args.data_path,
                                joint_transform=None,
                                transform=input_transform,
                                target_transform=target_transform)
        val_loader = DataLoader(val_set,
                                batch_size=4,
                                num_workers=2, shuffle=False)
    elif 'cityscapes' in args.dataset:
        val_set = cityscapes.CityScapes('fine', 'val',
                                        data_path=args.data_path,
                                        joint_transform=None,
                                        transform=input_transform,
                                        target_transform=target_transform)
        val_loader = DataLoader(val_set,
                                batch_size=args.val_batch_size,
                                num_workers=2, shuffle=False)
    elif 'msdHeart' in args.dataset:
        val_set = msdHeart.MSD_Heart('fine', 'test' if test else 'val',
                                data_path=args.data_path,
                                code_path=args.code_path, 
                                joint_transform=None,
                                transform=input_transform,
                                target_transform=target_transform)
        val_loader = DataLoader(val_set,
                                batch_size=4,
                                num_workers=2, shuffle=False)
    elif 'acdc' in args.dataset:
        val_set = acdc.ACDC('fine', 'test' if test else 'val',
                                data_path=args.data_path,
                                code_path=args.code_path, 
                                joint_transform=None,
                                transform=input_transform,
                                target_transform=target_transform)
        val_loader = DataLoader(val_set,
                                batch_size=4,
                                num_workers=2, shuffle=False)
    elif 'brats18' in args.dataset:
        if args.modality == '2D':
            val_set = brats18_2D.BraTS18_2D('fine', 'test' if test else 'val',
                                data_path=args.data_path,
                                code_path=args.code_path, 
                                joint_transform=None,
                                transform=input_transform,
                                target_transform=target_transform)
            val_loader = DataLoader(val_set,
                                    batch_size=4,
                                    num_workers=2, shuffle=False)
        else: #modality == 3D
            val_set = brats18.Brats18(mode='val', data_path=args.data_path,
                                            code_path=args.code_path, subset = False)
            val_loader = DataLoader(val_set,
                                    batch_size=args.val_batch_size,
                                    num_workers=2, shuffle=False)
    else: 
        print("Specify dataset!!!")

    print('Starting test...')
    if args.modality == '2D':
        vl_loss, val_acc, val_iu, iu_xclass = test(val_loader, net, criterion)
        ## Append info to logger
        info = [vl_loss, val_acc, val_iu]

        if 'brats18' or 'acdc' in args.dataset:
            log_columns = ['Val loss, ', 'Val Acc, ', 'Mean dice, ', 'Mean dice cl0, ', 'Mean dice cl1, ', 'Mean dice cl2, ', 'Mean dice cl3']

    else:   # modality == '3D'
        vl_loss, val_acc, val_iu, iu_xclass, meanDice, meanDiceWT, meanDiceTC, meanDiceET = test(val_loader, net, criterion)
        ## Append info to logger
        info = [vl_loss, val_acc, meanDice, meanDiceWT, meanDiceTC, meanDiceET]

        log_columns = ['Val loss, ', 'Val Acc, ', 'Mean dice, ', 'Mean dice WT, ', 'Mean dice TC, ', 'Mean dice ET']
    
    if args.modality == '2D':
        for cl in range(val_loader.dataset.num_classes):
            info.append(iu_xclass[cl])
    rew_log = open(os.path.join(args.ckpt_path, args.exp_name, 'test_results.txt'), 'a')

    if args.modality == '3D' or args.dataset == 'brats18' or args.dataset == 'acdc':
        for col in log_columns:
            rew_log.write(col)
        rew_log.write("\n")


    for inf in info:
        rew_log.write("%f," % (inf))
    rew_log.write("\n")
