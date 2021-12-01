import os
import sys
import shutil
import numpy as np
import random
import math

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, MultiStepLR

from models.model_utils import create_models, load_models
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, train, validate, final_test
import utils.parser as parser

import torch.nn.functional as F
#from torchgeometry.losses import 

cudnn.benchmark = False
cudnn.deterministic = True


def main(args):
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ####------ Create experiment folder  ------####
    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))

    ####------ Print and save arguments in experiment folder  ------####
    parser.save_arguments(args)
    ####------ Copy current config file to ckpt folder ------####
    fn = sys.argv[0].rsplit('/', 1)[-1]
    shutil.copy(sys.argv[0], os.path.join(args.ckpt_path, args.exp_name, fn))

    print("torch.cuda.is_available()", torch.cuda.is_available())

    ####------ Create segmentation, query and target networks ------####
    kwargs_models = {"dataset": args.dataset,
                     "al_algorithm": args.al_algorithm,
                     "region_size": args.region_size
                     }
    net, _, _ = create_models(**kwargs_models)
    #print(net)
    ####------ Load weights if necessary and create log file ------####
    kwargs_load = {"net": net,
                   "load_weights": args.load_weights,
                   "exp_name_toload": args.exp_name_toload,
                   "snapshot": args.snapshot,
                   "exp_name": args.exp_name,
                   "ckpt_path": args.ckpt_path,
                   "checkpointer": args.checkpointer,
                   "exp_name_toload_rl": args.exp_name_toload_rl,
                   "policy_net": None,
                   "target_net": None,
                   "test": args.test,
                   "dataset": args.dataset,
                   "al_algorithm": 'None'}
    logger, curr_epoch, best_record = load_models(**kwargs_load)
    #print(net)
    ####------ Load training and validation data ------####
    kwargs_data = {"data_path": args.data_path,
                   "code_path": args.code_path,
                   "tr_bs": args.train_batch_size,
                   "vl_bs": args.val_batch_size,
                   "n_workers": 4,
                   "scale_size": args.scale_size,
                   "input_size": args.input_size,
                   "num_each_iter": args.num_each_iter,
                   "only_last_labeled": args.only_last_labeled,
                   "dataset": args.dataset,
                   "test": args.test,
                   "al_algorithm": args.al_algorithm,
                   "full_res": args.full_res,
                   "region_size": args.region_size,
                   "supervised": True}

    train_loader, _, val_loader, _ = get_data(**kwargs_data)

    ####------ Create losses ------####
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.ignore_label).cuda()
    #import monai.losses as losses
    #criterion = losses.Dice()

    # class MyCrossEntropy(nn.CrossEntropyLoss):
    #     def forward(self, input, target):
    #         target = target.long()
    #         return F.cross_entropy(input, target, weight=self.weight, ignore_index=train_loader.dataset.ignore_label, reduction=self.reduction).cuda()

    # criterion = MyCrossEntropy()
    ####------ Create optimizers (and load them if necessary) ------####
    kwargs_load_opt = {"net": net,
                       "opt_choice": args.optimizer,
                       "lr": args.lr,
                       "wd": args.weight_decay,
                       "momentum": args.momentum,
                       "ckpt_path": args.ckpt_path,
                       "exp_name_toload": args.exp_name_toload,
                       "exp_name": args.exp_name,
                       "snapshot": args.snapshot,
                       "checkpointer": args.checkpointer,
                       "load_opt": args.load_opt,
                       "policy_net": None,
                       "lr_dqn": args.lr_dqn,
                       "al_algorithm": 'None'}

    optimizer, _ = create_and_load_optimizers(**kwargs_load_opt)

    # Early stopping params initialization
    es_val = 0
    es_counter = 0
    es_dice = 0

    if args.train:
        print('Starting training...')
        #@carina scheduler off
        scheduler = ExponentialLR(optimizer, gamma=0.998)
        net.train()

        if args.modality == '2D':
            for epoch in range(curr_epoch, args.epoch_num + 1):
                print('Epoch %i /%i' % (epoch, args.epoch_num + 1))
                tr_loss, _, tr_acc, tr_iu = train(train_loader, net, criterion,
                                                optimizer, supervised=True)

                if epoch % 1 == 0:
                    vl_loss, val_acc, val_iu, iu_xclass, _ = validate(val_loader, net, criterion,
                                                                    optimizer, epoch, best_record,
                                                                    args)

                ## Append info to logger
                info = [epoch, optimizer.param_groups[0]['lr'],
                        tr_loss,
                        0, vl_loss, tr_acc, val_acc, tr_iu, val_iu]
                for cl in range(train_loader.dataset.num_classes):
                    info.append(iu_xclass[cl])
                logger.append(info)
                #@carina scheduler off
                scheduler.step()
                # Early stopping with val jaccard
                es_counter += 1
                if val_iu > es_val and not math.isnan(val_iu):
                    torch.save(net.cpu().state_dict(),
                            os.path.join(args.ckpt_path, args.exp_name,
                                            'best_jaccard_val.pth'))
                    net.cuda()
                    es_val = val_iu
                    es_counter = 0
                elif es_counter > args.patience:
                    print('Patience for Early Stopping reached!')
                    break

            logger.close()
        
        # 3D brats
        else:
            for epoch in range(curr_epoch, args.epoch_num + 1):
                print('Epoch %i /%i' % (epoch, args.epoch_num + 1))
                # adapt train loss,, ...
                # tr_loss, _, meanDice, meanDiceWT, meanDiceTC, meanDiceET = train(train_loader, net, criterion,
                #                                 optimizer, supervised=True)
                tr_loss, _, tr_acc, tr_iu, tr_meanDice = train(train_loader, net, criterion,
                                                optimizer, supervised=True)

                if epoch % 1 == 0:
                    vl_loss, val_acc, val_iu, iu_xclass, meanDice, meanDiceWT, meanDiceTC, meanDiceET, _ = validate(val_loader, net, criterion,
                                                                                                            optimizer, epoch, best_record, args)
                                                                
                ## Append info to logger
                info = [epoch, optimizer.param_groups[0]['lr'],
                        tr_loss, tr_acc, tr_meanDice, vl_loss, val_acc, meanDice, meanDiceWT, meanDiceTC, meanDiceET]
                logger.append(info)
                #@carina scheduler off
                scheduler.step()
                # Early stopping with val jaccard
                es_counter += 1
                if meanDice > es_dice and not math.isnan(meanDice):
                    torch.save(net.cpu().state_dict(),
                            os.path.join(args.ckpt_path, args.exp_name,
                                            'best_jaccard_val.pth'))
                    net.cuda()
                    es_dice = meanDice
                    es_counter = 0
                elif es_counter > args.patience:
                    print('Patience for Early Stopping reached!')
                    break

                del (tr_loss, tr_acc, tr_iu, tr_meanDice, vl_loss, val_acc, meanDice, meanDiceWT, meanDiceTC, meanDiceET)

            logger.close()

    if args.final_test:
        final_test(args, net, criterion)


if __name__ == '__main__':
    ####------ Parse arguments from console  ------####
    print("torch.cuda.is_available()", torch.cuda.is_available())
    #gpu=1
    #device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    #if torch.cuda.is_available():
    #    torch.cuda.set_device(device)
    #torch.cuda.set_device(1)
    torch.cuda.empty_cache() 
    import sys
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION') # 1.9.0+cu102
    from subprocess import call
    #torch.cuda.set_device(0)
    # call(["nvcc", "--version"]) does not work
    print('__CUDNN VERSION:', torch.backends.cudnn.version()) #7605
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())

    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())

    print("torch.cuda.is_available()", torch.cuda.is_available())
    print("torch.version.cuda", torch.version.cuda)
    args = parser.get_arguments()
    main(args)
    torch.cuda.empty_cache() 

