#from models.noNewNet import NoNewNet
import time
import math
import numpy as np
import os
import random
from numpy.lib.type_check import imag
from scipy.stats import entropy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.query_network import QueryNetworkDQN
from models.fpn import FPN50
from models.fpn_bayesian import FPN50_bayesian

from models.noNewNet import NoNewNet

from utils.final_utils import get_logfile
from utils.progressbar import progress_bar
import utils.parser as parser

from memory_profiler import profile

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10

n_cl_brats18 = 4

def create_models(dataset, al_algorithm, region_size):
    """Returns segmentation model FPN with backbone ResNet50
    :param dataset: (str) Which dataset to use.
    :param al_algorithm: (str) Which active learning algorithm to use: 'random', 'entropy', 'bald', 'ralis'
    :param region_size: (tuple) Size of regions, [width, height]
    :return: segmentation network, query network, target network (same construction as query network)
    """
    args = parser.get_arguments()
    if args.modality == '2D':       
        # Segmentation network
        #n_cl = 11 if 'camvid' in dataset else 19
        if 'camvid' in dataset:
            n_cl = 11
        elif 'cityscapes' in dataset:
            n_cl = 19
        elif 'acdc' in dataset:
            n_cl = 4
        elif 'msdHeart' in dataset:
            n_cl = 2
        elif 'brats18' in dataset:
            n_cl = n_cl_brats18
        else:
            print('Specify the correct number of classes of your dataset')
        net_type = FPN50_bayesian if al_algorithm == 'bald' else FPN50
        # if 'BraTS18' in dataset:
        #     net_type = NoNewNet_
        net = net_type(num_classes=n_cl).cuda()
        print('Model has ' + str(count_parameters(net)) + ' parameters.')

        # Query network (and target network for DQN)
        if "agnostic" in args.exp_name:
            #input_size = [3 * 64, 3 * 64]  #here for ACDC [192, 192]
            input_size = [3 * region_size[0], 3 * region_size[1]]
        else:
            if 'brats18' in dataset:  
                #input_size = [(n_cl + 1) + 3 * 64, (n_cl + 1) + 3 * 64] #[197,197]
                input_size = [167, 167] #[125, 149] RuntimeError: running_mean should contain 167 elements not 197
                print("input_size: ", input_size)
            else:
                input_size = [(n_cl + 1) + 3 * 64, (n_cl + 1) + 3 * 64]  #here for ACDC [197, 197]
                #input_size = [(n_cl + 1) + 3 * region_size[0], (n_cl + 1) + 3 * region_size[0]]
                print("input_size to network: ", input_size)
        if al_algorithm == 'ralis':
            #image_size = [480, 360] if 'camvid' or 'acdc' in dataset else [2048, 1024]
            if 'acdc' in dataset:
                image_size = [256, 256]
            elif 'camvid' in dataset:
                image_size = [480, 360]
            elif 'msdHeart' in dataset:
                image_size = [320, 320] 
            elif 'brats18' in dataset:
                image_size = [160, 192] #oder [128,128] or [160, 192]
            else:
                image_size = [2048, 1024]
                print('Check the image size for dataset!')
            #@carina changed indexes_full_state for ACDC
            if 'acdc' in dataset:
                print("indexes_full_state for acdc")
                if region_size[0] == 32:        #9 * 8 * 8 = 576
                    indexes_full_state = 18 * (image_size[0] // region_size[0]) * (image_size[1] // region_size[1]) #2560=160*((128/32)(128/32)), 288 for 18*(256/64)*(256/64)
                elif region_size[0] == 64:
                    indexes_full_state = 18 * (image_size[0] // region_size[0]) * (image_size[1] // region_size[1]) #2560=160*((128/32)(128/32)), 288 for 18*(256/64)*(256/64)
                else:
                    print("DEFINE indexes full state! ")
                    indexes_full_state = 18 * (image_size[0] // region_size[0]) * (image_size[1] // region_size[1]) #2560=160*((128/32)(128/32)), 288 for 18*(256/64)*(256/64)
                
            # elif 'msdHeart' in dataset:
            #     print("indexes_full_state for msdHeart")
            #     indexes_full_state = 3 * (image_size[0] // region_size[0]) * (image_size[1] // region_size[1])
            elif 'brats18' in dataset:
                print("check indexes_full_state")
                indexes_full_state = 160 * (image_size[0] // region_size[0]) * (image_size[1] // region_size[1]) # (160/40)*(192/48)=16 
                # running_mean should contain 2560 elements not 288
                print("indexes full state: ", indexes_full_state)
            else:
                indexes_full_state = 10 * (image_size[0] // region_size[0]) * (image_size[1] // region_size[1]) #here: 160 for ACDC

            print("indexes_full_state for ", dataset, " = ", str(indexes_full_state))
            policy_net = QueryNetworkDQN(input_size=input_size[0], input_size_subset=input_size[1],
                                        indexes_full_state=indexes_full_state).cuda()
            target_net = QueryNetworkDQN(input_size=input_size[0], input_size_subset=input_size[1],
                                        indexes_full_state=indexes_full_state).cuda()
            #print("policy_net: ", policy_net)
            print('Policy network has ' + str(count_parameters(policy_net)))
        else:
            policy_net = None
            target_net = None
        print('Models created!')

    ################ 3D modality:#########################################################################################
    else:
         # Segmentation network
        #n_cl = 11 if 'camvid' in dataset else 19
        if 'brats18' in dataset:
            n_cl = n_cl_brats18
        else:
            print('Specify the correct number of classes of your dataset')
        # net_type = NoNewNet
        net = NoNewNet().cuda() #@carina
        # net = net_type(num_classes=n_cl).cuda()
        print('Model has ' + str(count_parameters(net)))

        # Query network (and target network for DQN)
        if "agnostic" in args.exp_name:
            input_size = [3 * 64, 3 * 64]  #for ACDC [192, 192]
        else:
            #input_size = [(n_cl + 1) + 3 * 64, (n_cl + 1) + 3 * 64]  #for ACDC [198, 198]
            input_size = [(n_cl + 1) + 3 * region_size[0], (n_cl + 1) + 3 * region_size[1]]
            print("input size: ", input_size)
        if al_algorithm == 'ralis':
            #image_size = [480, 360] if 'camvid' or 'acdc' in dataset else [2048, 1024]
            if 'brats18' in dataset:
                image_size = [128, 128, 128]
            else:
                print("Specify correct image size")
            if 'brats18' in dataset:
                print("indexes_full_state for brats 18 ")
                indexes_full_state = 18 * (image_size[0] // region_size[0]) * (image_size[1] // region_size[1]) #288 for 18*(256/64)*(256/64)
            else:
                indexes_full_state = 10 * (image_size[0] // region_size[0]) * (image_size[1] // region_size[1]) #here: 160 for ACDC
            print("indexes_full_state for ", dataset, " = ", str(indexes_full_state))
            policy_net = QueryNetworkDQN(input_size=input_size[0], input_size_subset=input_size[1],
                                        indexes_full_state=indexes_full_state).cuda()
            target_net = QueryNetworkDQN(input_size=input_size[0], input_size_subset=input_size[1],
                                        indexes_full_state=indexes_full_state).cuda()
            #print("policy_net: ", policy_net)
            print('Policy network has ' + str(count_parameters(policy_net)))
        else:
            policy_net = None
            target_net = None
        print('Models created!') 
    return net, policy_net, target_net


def count_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def load_models(net, load_weights, exp_name_toload, snapshot,
                exp_name, ckpt_path, checkpointer, exp_name_toload_rl='',
                policy_net=None, target_net=None, test=False, dataset='cityscapes', al_algorithm='random'):
    """Load model weights.
    :param net: Segmentation network
    :param load_weights: (bool) True if segmentation network is loaded from pretrained weights in 'exp_name_toload'
    :param exp_name_toload: (str) Folder where to find pre-trained segmentation network's weights.
    :param snapshot: (str) Name of the checkpoint.
    :param exp_name: (str) Experiment name.
    :param ckpt_path: (str) Checkpoint name.
    :param checkpointer: (bool) If True, load weights from the same folder.
    :param exp_name_toload_rl: (str) Folder where to find trained weights for the query network (DQN). Used to test
    query network.
    :param policy_net: Policy network.
    :param target_net: Target network.
    :param test: (bool) If True and al_algorithm='ralis' and there exists a checkpoint in 'exp_name_toload_rl',
    we will load checkpoint for trained query network (DQN).
    :param dataset: (str) Which dataset.
    :param al_algorithm: (str) Which active learning algorithm.
    """
    args = parser.get_arguments()
    policy_path = os.path.join(ckpt_path, exp_name_toload_rl, 'policy_' + snapshot)
    net_path = os.path.join(ckpt_path, exp_name_toload, 'best_jaccard_val.pth')

    policy_checkpoint_path = os.path.join(ckpt_path, exp_name, 'policy_' + snapshot)
    net_checkpoint_path = os.path.join(ckpt_path, exp_name, 'last_jaccard_val.pth')

    if args.modality == '2D':
        ####------ Load policy (RL) from one folder and network from another folder ------####
        if al_algorithm == 'ralis' and test and os.path.isfile(policy_path):
            print ('(RL and TEST) Testing policy from another experiment folder!')
            policy_net.load_state_dict(torch.load(policy_path))
            # Load pre-trained segmentation network from another folder (best checkpoint)
            if load_weights and len(exp_name_toload) > 0:
                print('Loading pre-trained segmentation network (best checkpoint).')
                net_dict = torch.load(net_path)
                if len([key for key, value in net_dict.items() if 'module' in key.lower()]) > 0:
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in net_dict.items():
                        name = k[7:]  # remove module.
                        new_state_dict[name] = v
                    net_dict = new_state_dict
                
                # load model except for last layer @carina
                # pre-trained on one dataset, train on other
                if 'camvid' in exp_name_toload:
                    n_cl = 11 #camvid 
                elif 'msdHeart' in exp_name_toload:
                    n_cl = 2
                elif 'acdc' in exp_name_toload:
                    n_cl = 4
                elif 'cityscapes' in exp_name_toload:
                    n_cl = 19
                elif 'brats18' in exp_name_toload:
                    n_cl = n_cl_brats18
                else: 
                    print("Check number of classes of trained segmentation model")

                net_type = FPN50_bayesian if al_algorithm == 'bald' else FPN50
                net = net_type(num_classes=n_cl).cuda()
                #print("net before seg loading net.layer1: ", net.layer1.weight)
                net.load_state_dict(net_dict)
                #print("net after seg loading net.layer1: ", net.layer1.weight)
                #re-initialise last layer to new number of classes in dataset
                if 'camvid' in dataset:
                    n_cl = 11
                elif 'acdc' in dataset:
                    n_cl = 4
                elif 'msdHeart' in dataset:
                    n_cl = 2
                elif 'cityscapes' in dataset:
                    n_cl = 19
                elif 'brats18' in dataset:
                    n_cl = n_cl_brats18
                else:
                    print("Define the correct number of output nodes!")
                net.classifier = torch.nn.Conv2d(in_channels=512, out_channels=n_cl, kernel_size=3, stride=1, padding=1)
                #net.load_state_dict(net_dict)
                net.cuda()

            if checkpointer:  # In case the experiment is interrupted, load most recent segmentation network
                if os.path.isfile(net_checkpoint_path):
                    print('(RL and TEST) Loading checkpoint for segmentation network!')
                    net_dict = torch.load(net_checkpoint_path)
                    if len([key for key, value in net_dict.items() if 'module' in key.lower()]) > 0:
                        from collections import OrderedDict
                        new_state_dict = OrderedDict()
                        for k, v in net_dict.items():
                            name = k[7:]  # remove module.
                            new_state_dict[name] = v
                        net_dict = new_state_dict
                    net.load_state_dict(net_dict)


        else:
            ####------ Load experiment from another folder ------####
            if load_weights and len(exp_name_toload) > 0:
                print('(From another exp) training resumes from best_jaccard_val.pth')
                net_dict = torch.load(net_path)
                if len([key for key, value in net_dict.items() if 'module' in key.lower()]) > 0:
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in net_dict.items():
                        name = k[7:]  # remove module.
                        new_state_dict[name] = v
                    net_dict = new_state_dict
                #net.load_state_dict(net_dict)

                # load model except for last layer @carina
                # pre-trained on one dataset, train on other
                if 'camvid' in exp_name_toload:
                    n_cl = 11 #camvid 
                elif 'msdHeart' in exp_name_toload:
                    n_cl = 2
                elif 'acdc' in exp_name_toload:
                    n_cl = 4
                elif 'brats18' in dataset:
                    n_cl = 4
                elif 'cityscapes' in exp_name_toload:
                    n_cl = 19
                else: 
                    print("Check number of classes of trained segmentation model")

                net_type = FPN50_bayesian if al_algorithm == 'bald' else FPN50
                net = net_type(num_classes=n_cl).cuda()
                net.load_state_dict(net_dict)

                #re-initialise last layer to new number of classes in dataset
                if 'camvid' in dataset:
                    n_cl = 11
                elif 'acdc' in dataset:
                    n_cl = 4
                elif 'brats18' in dataset:
                    n_cl = 4
                elif 'msdHeart' in dataset:
                    n_cl = 2
                elif 'cityscapes' in dataset:
                    n_cl = 19
                else:
                    print("Define the correct number of output nodes!")
                net.classifier = torch.nn.Conv2d(in_channels=512, out_channels=n_cl, kernel_size=3, stride=1, padding=1)

                # change hard-coded
                # n_cl = 11 
                # net_type = FPN50_bayesian if al_algorithm == 'bald' else FPN50 #_preTrain #@carina
                # net = net_type(num_classes=n_cl).cuda()   

                # n_cl = 4
                # net.classifier = torch.nn.Conv2d(in_channels=512, out_channels=n_cl, kernel_size=3, stride=1, padding=1)
                

            ####------ Resume experiment ------####
            if checkpointer:
                ##-- Check if weights exist --##
                if os.path.isfile(net_checkpoint_path):
                    print('(Checkpointer) training resumes from last_jaccard_val.pth')
                    net_dict = torch.load(net_checkpoint_path)
                    if len([key for key, value in net_dict.items() if 'module' in key.lower()]) > 0:
                        from collections import OrderedDict
                        new_state_dict = OrderedDict()
                        for k, v in net_dict.items():
                            name = k[7:]  # remove module.
                            new_state_dict[name] = v
                        net_dict = new_state_dict
                    net.load_state_dict(net_dict)

                    if al_algorithm == 'ralis' and os.path.isfile(policy_checkpoint_path) \
                            and policy_net is not None and target_net is not None:
                        print('(Checkpointer RL) training resumes from ' + snapshot)
                        policy_net.load_state_dict(torch.load(policy_checkpoint_path))
                        policy_net.cuda()
                        target_net.load_state_dict(torch.load(policy_checkpoint_path))
                        target_net.cuda()
                else:
                    print('(Checkpointer) Training starts from scratch')

        ####------ Get log file ------####
        if al_algorithm == 'ralis':
            logger = None
            best_record = None
            curr_epoch = None
        else:
            #num_classes = 11 if 'camvid' in dataset else 19
            if 'camvid' in dataset:
                num_classes = 11
            elif 'acdc' in dataset:
                num_classes = 4
            elif 'msdHeart' in dataset:
                num_classes = 2
            elif 'cityscapes' in dataset:
                num_classes = 19
            elif 'brats18' in dataset:
                num_classes = n_cl_brats18
            else:
                print("Define the correct number of classes for your dataset! ")
            logger, best_record, curr_epoch = get_logfile(ckpt_path, exp_name, checkpointer, snapshot,
                                                        num_classes=num_classes)

    # 3D modality
    else:
        print("Load models for Brats18")
        ####------ Load policy (RL) from one folder and network from another folder ------####
        if al_algorithm == 'ralis' and test and os.path.isfile(policy_path):
            print ('(RL and TEST) Testing policy from another experiment folder!')
            policy_net.load_state_dict(torch.load(policy_path))
            # Load pre-trained segmentation network from another folder (best checkpoint)
            if load_weights and len(exp_name_toload) > 0:
                print('Loading pre-trained segmentation network (best checkpoint).')
                net_dict = torch.load(net_path)
                if len([key for key, value in net_dict.items() if 'module' in key.lower()]) > 0:
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in net_dict.items():
                        name = k[7:]  # remove module.
                        new_state_dict[name] = v
                    net_dict = new_state_dict
                net.load_state_dict(net_dict)
                net.cuda()

            if checkpointer:  # In case the experiment is interrupted, load most recent segmentation network
                if os.path.isfile(net_checkpoint_path):
                    print('(RL and TEST) Loading checkpoint for segmentation network!')
                    net_dict = torch.load(net_checkpoint_path)
                    if len([key for key, value in net_dict.items() if 'module' in key.lower()]) > 0:
                        from collections import OrderedDict
                        new_state_dict = OrderedDict()
                        for k, v in net_dict.items():
                            name = k[7:]  # remove module.
                            new_state_dict[name] = v
                        net_dict = new_state_dict
                    net.load_state_dict(net_dict)
        else:
            print("specify for baselines 3D")

        ####------ Get log file ------####
        if al_algorithm == 'ralis':
            logger = None
            best_record = None
            curr_epoch = None
        else:
            num_classes = n_cl_brats18 if 'brats18' in dataset else print("specify correct number of classes! ")
            logger, best_record, curr_epoch = get_logfile(ckpt_path, exp_name, checkpointer, snapshot,
                                                        num_classes=num_classes)

    return logger, curr_epoch, best_record


def get_region_candidates(candidates, train_set, num_regions=2):
    """Get region candidates function.
    :param candidates: (list) randomly sampled image indexes for images that contain unlabeled regions.
    :param train_set: Training set.
    :param num_regions: Number of regions to take as possible regions to be labeled.
    :return: candidate_regions: List of tuples (int(Image index), int(width_coord), int(height_coord)).
        The coordinate is the left upper corner of the region.
    """
    args = parser.get_arguments()
    if args.modality == '2D':
        s = time.time()
        print('Getting region candidates...')
        total_regions = num_regions
        candidate_regions = []
        #### --- Get candidate regions --- ####
        counter_regions = 0
        available_regions = train_set.get_num_unlabeled_regions()
        print ('Available regions are ' + str(available_regions)) #@carina 
        rx, ry = train_set.get_unlabeled_regions()
        while counter_regions < total_regions and (total_regions - counter_regions) <= available_regions:
            index_ = np.random.choice(len(candidates))
            index = candidates[index_]
            num_regions_left = train_set.get_num_unlabeled_regions_image(int(index))
            if num_regions_left > 0:
                counter_x, counter_y = train_set.get_random_unlabeled_region_image(int(index))
                candidate_regions.append((int(index), counter_x, counter_y))
                available_regions -= 1
                counter_regions += 1
                if num_regions_left == 1:
                    candidates.pop(int(index_))
            else:
                print ('This image has no more unlabeled regions!')

        train_set.set_unlabeled_regions(rx, ry)
        print ('Regions candidates indexed! Time elapsed: ' + str(time.time() - s))
        print ('Candidate regions are ' + str(counter_regions))
        print ('Candidate regions are len(candidate_regions) ' + str(len(candidate_regions))) #@carina 
        return candidate_regions
   
    else:  # 3D 
        s = time.time()
        print('Getting region candidates...')
        total_regions = num_regions
        candidate_regions = []
        #### --- Get candidate regions --- ####
        counter_regions = 0
        available_regions = train_set.get_num_unlabeled_regions()
        print ('Available regions are ' + str(available_regions)) #@carina 
        rx, ry, rz = train_set.get_unlabeled_regions()
        while counter_regions < total_regions and (total_regions - counter_regions) <= available_regions:
            index_ = np.random.choice(len(candidates))
            index = candidates[index_]
            num_regions_left = train_set.get_num_unlabeled_regions_image(int(index))
            if num_regions_left > 0:
                counter_x, counter_y, counter_z = train_set.get_random_unlabeled_region_image(int(index))
                candidate_regions.append((int(index), counter_x, counter_y, counter_z))
                available_regions -= 1
                counter_regions += 1
                if num_regions_left == 1:
                    candidates.pop(int(index_))
            else:
                print ('This image has no more unlabeled regions!')

        train_set.set_unlabeled_regions(rx, ry, rz)
        print ('Regions candidates indexed! Time elapsed: ' + str(time.time() - s))
        print ('Candidate regions are ' + str(counter_regions))
        print ('Candidate regions are len(candidate_regions) ' + str(len(candidate_regions))) #@carina 
        return candidate_regions


#@profile
def compute_state(args, net, region_candidates, candidate_set, train_set, num_groups=5, reg_sz=(128, 128)):
    """Computes the state of a given candidate pool of size N.
    :(argparse.ArgumentParser) args: The parser with all the defined arguments.
    :param net: Segmentation network.
    :param region_candidates: (list) of region candidates to be potentially labeled in this iteration
    :param candidate_set: (list) of image indexes to be potentially labeled in this iteration
    :param train_set: Training dataset
    :param num_groups: (int) Number of regions to label at this iteration
    :param reg_sz: (tuple) region size.
    :return: a Torch tensor with the state-action representation, region candidates rearranged to match
    the order in state-action representation
    """
    cand = 0
    s = time.time()
    print('Computing state...')
    net.eval()
    state = []
    old_candidate = None
    predictions_py = None
    predictions_py_prob = None
    pred_py = None
    ent = None
    bald_map = None

    if args.al_algorithm == 'bald':
        net.apply(apply_dropout)

    region_candidates.sort()

    #  List of tuples (int(Image index), int(width_coord), int(height_coord), int(depth_coord)).
    for candidates in region_candidates:
        # cadidates[0] is image index
        if not candidates[0] == old_candidate:
            #print('candidates[0]: ', candidates[0])
            #print('old_candidate: ', old_candidate)
            del (pred_py)
            del (predictions_py_prob)
            del (predictions_py)
            if args.al_algorithm in ['entropy', 'ralis']:
                del (ent)
            if args.al_algorithm == 'bald':
                del (bald_map)
            old_candidate = candidates[0]
            inputs, gts, _, _ = candidate_set.get_specific_item(candidates[0])
            inputs, gts = Variable(inputs).cuda(), Variable(gts).cuda()
            if args.al_algorithm == 'bald':
                outputs, _ = net(inputs.unsqueeze(0), bayesian=True, n_iter=args.bald_iter) 
                outputs = torch.cat(outputs)
            else:
                #print('inputs: ', inputs.shape)
                outputs, _ = net(inputs.unsqueeze(0)) 

            pred_py = F.softmax(outputs, dim=1).data
            del (outputs)
            if args.al_algorithm == 'bald':
                bald_map, pred_py = compute_bald(pred_py.cpu())
                predictions_py = pred_py.squeeze_(1).cpu().type(torch.FloatTensor)
                predictions_py_prob = None
            else:
                pred_py = pred_py.max(1)
                predictions_py = pred_py[1].squeeze_(1).cpu().type(torch.FloatTensor)  #prediction of classes for each pixel
                #print('shape of predictions_py: ', predictions_py.shape)  # torch.Size([1, 256, 256])
                predictions_py_prob = pred_py[0].squeeze_(1).cpu().type(torch.FloatTensor) #probability for each prediction
                #print('shape of predictions_py_prob: ', predictions_py_prob.shape)
            if args.al_algorithm in ['entropy', 'ralis']:
                ent = compute_entropy_seg(args, inputs, net)
                print('shape of entropy : ', ent.shape)

        if args.al_algorithm == 'bald':
            bald_region = bald_map[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                          int(candidates[1]):int(candidates[1]) + reg_sz[0]]
            bald_region = bald_region.mean()
            pred_region = predictions_py[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                          int(candidates[1]):int(candidates[1]) + reg_sz[0]]
            pred_region_prob = None
        else:
            pred_region = predictions_py[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                          int(candidates[1]):int(candidates[1]) + reg_sz[0]]
            pred_region_prob = predictions_py_prob[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                               int(candidates[1]):int(candidates[1]) + reg_sz[0]]

        if args.al_algorithm == 'entropy':
            ent_region = ent[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                         int(candidates[1]):int(candidates[1]) + reg_sz[0]].mean()
            sample_stats = []
            sample_stats.append(ent_region.item())
        # Adding action representation
        elif args.al_algorithm == 'ralis':
            if 'brats18' in args.dataset: #@carina
                ent_region = ent[0, int(candidates[1]):int(candidates[1]) + reg_sz[0],  #ent.shape = torch.Size([1, 256, 256])
                            int(candidates[2]):int(candidates[2]) + reg_sz[1]]
            else:
                ent_region = ent[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],  #ent.shape = torch.Size([1, 256, 256])
                            int(candidates[1]):int(candidates[1]) + reg_sz[0]]
            sample_stats = create_feature_vector_3H_region_kl_sim(pred_region, ent_region, train_set,
                                                                num_classes=train_set.num_classes, reg_sz=reg_sz)
        elif args.al_algorithm == 'random':
            sample_stats = []
            sample_stats.append(0.0)
        elif args.al_algorithm == 'bald':
            sample_stats = []
            sample_stats.append(bald_region.item())
        # unsqueeze returns a new tensor with a dimension of size one inserted at the specified position
        # tensor.detach() creates a tensor that shares storage with tensor that does not require grad. It detaches the output from the computational graph. 
        # So no gradient will be backpropagated along this variable.
        # The wrapper with torch.no_grad() temporarily set all the requires_grad flag to false. torch.no_grad says that no operation should build the graph
        state.append(torch.Tensor(sample_stats).unsqueeze(0).detach())

        del (pred_region)
        del (pred_region_prob)
        if args.al_algorithm in ['entropy', 'ralis']:
            del (ent_region)
        if args.al_algorithm == 'bald':
            del (bald_region)

    # end of for loop over candidates
    del (pred_py)
    del (predictions_py)
    del (predictions_py_prob)

    #if not cand == 256:
    #print("state.shape: ", len(state))
    state = torch.cat(state, dim=0) # Concatenates the given sequence of seq tensors in the given dimension.

    # Shuffle vector so that we do not end up with regions from the same image in the same pool
    # for candidates[2]==256: randperm=tensor([0, 3, 2, 1])
    randperm = torch.randperm(state.size()[0])
    state = state[randperm]
    region_candidates = np.array(region_candidates)[randperm]
    #print('region_candidates.shape: ', len(region_candidates))
    state = state.view(num_groups, state.size()[0] // num_groups, state.size()[1])
    #print('state.shape: ', state.size())
    region_candidates = np.reshape(region_candidates,
                                (num_groups, region_candidates.shape[0] // num_groups, region_candidates.shape[1]))
    state = Variable(state).cpu()  # [groups, cand_regions, channels, reg_size, reg_size]

    if args.al_algorithm == 'ralis':
        # Adding KL terms for action representation
        state = add_kl_pool2(state, n_cl=train_set.num_classes)
        # Add fixed part of state (from state subset)
        state_subset = []
        for index in range(0, len(candidate_set.state_subset)):  #for brats2d: len is 640
            # len of candidate set is 6240
            inputs, gts, _, _, regions = candidate_set.get_subset_state(index) #inputs: torch.Size([3, 256, 256]) / for brats: [3,160,192]
            # gts ([160, 192])
            inputs, gts = Variable(inputs).cuda(), Variable(gts).cuda()
            outputs, _ = net(inputs.unsqueeze(0)) #outputs = torch.Size([1, 4, 256, 256])
            #print("outputs.shape: ", outputs.shape)
            pred_py = F.softmax(outputs).data # pred_py: torch.Size([1, 4, 256, 256])
            #print("pred_py.shape: ", pred_py.shape)
            del (outputs)
            pred_py = pred_py.max(1) #.max(1) takes the max over dimension 1 and returns two tensors (max value in each row of pred_py, column index where max value is found).
            predictions_py = pred_py[1].squeeze_(1).cpu().type(torch.FloatTensor) #pred_py[1]=max values in each row; gives predictions for each pixel (torch.Size([1,256,256]))
            predictions_py_prob = pred_py[0].squeeze_(1).cpu().type(torch.FloatTensor) #gives probability for each prediction

            ent = compute_entropy_seg(args, inputs, net)  # ent.shape torch.Size([1, 256, 256])
            for reg in regions:
                if 'brats18' in args.dataset: #@carina
                    pred_region_prob = predictions_py_prob[0, int(reg[0]):int(reg[0]) + reg_sz[0],   # shape [64,64]
                                   int(reg[1]):int(reg[1]) + reg_sz[1]]
                    pred_region = predictions_py[0, int(reg[0]):int(reg[0]) + reg_sz[0],
                                int(reg[1]):int(reg[1]) + reg_sz[1]]
                    ent_region = ent[0, int(reg[0]):int(reg[0]) + reg_sz[0], int(reg[1]):int(reg[1]) + reg_sz[1]]
                    #print("ent_region: ", ent_region.shape) #64,64
                else:
                    pred_region_prob = predictions_py_prob[0, int(reg[1]):int(reg[1]) + reg_sz[1],
                                   int(reg[0]):int(reg[0]) + reg_sz[0]]
                    pred_region = predictions_py[0, int(reg[1]):int(reg[1]) + reg_sz[1],
                                int(reg[0]):int(reg[0]) + reg_sz[0]]
                    ent_region = ent[0, int(reg[1]):int(reg[1]) + reg_sz[1], int(reg[0]):int(reg[0]) + reg_sz[0]]

                # Convert 2D maps into vector representations
                sample_stats = create_feature_vector_3H_region_kl(pred_region, ent_region,
                                                                  num_classes=train_set.num_classes,
                                                                  reg_sz=reg_sz)
                state_subset.append(torch.Tensor(sample_stats).unsqueeze(0).detach())
                del (pred_region)
                del (pred_region_prob)
                del (ent_region)

            del (ent)
            del (pred_py)
            del (predictions_py)
            del (predictions_py_prob)
        state_subset = torch.cat(state_subset, dim=0)
    else:
        state_subset = None
    all_state = {'subset': state_subset, 'pool': state}
    if args.al_algorithm == 'bald':
        net.eval()
    print ('State computed! Time elapsed: ' + str(time.time() - s))
    return all_state, region_candidates

def compute_state_dataset_agnostic(args, net, region_candidates, candidate_set, train_set, num_groups=5, reg_sz=(128, 128)):
    """Computes the state of a given candidate pool of size N.
    :(argparse.ArgumentParser) args: The parser with all the defined arguments.
    :param net: Segmentation network.
    :param region_candidates: (list) of region candidates to be potentially labeled in this iteration
    :param candidate_set: (list) of image indexes to be potentially labeled in this iteration
    :param train_set: Training dataset
    :param num_groups: (int) Number of regions to label at this iteration
    :param reg_sz: (tuple) region size.
    :return: a Torch tensor with the state-action representation, region candidates rearranged to match
    the order in state-action representation
    """
    cand = 0
    s = time.time()
    print ('Computing state class agnostic...')
    net.eval()
    state = []
    old_candidate = None
    predictions_py = None
    predictions_py_prob = None
    pred_py = None
    ent = None
    bald_map = None

    if args.al_algorithm == 'bald':
        net.apply(apply_dropout)

    region_candidates.sort()
    #print('initial region candidates: ', region_candidates)

    for candidates in region_candidates:
        if not candidates[0] == old_candidate:
            #print('candidates[0]: ', candidates[0])
            #print('old_candidate: ', old_candidate)
            del (pred_py)
            del (predictions_py_prob)
            del (predictions_py)
            if args.al_algorithm in ['entropy', 'ralis']:
                del (ent)
            if args.al_algorithm == 'bald':
                del (bald_map)
            old_candidate = candidates[0]
            inputs, gts, _, _ = candidate_set.get_specific_item(candidates[0])
            inputs, gts = Variable(inputs).cuda(), Variable(gts).cuda()
            if args.al_algorithm == 'bald':
                outputs, _ = net(inputs.unsqueeze(0), bayesian=True, n_iter=args.bald_iter) 
                outputs = torch.cat(outputs)
            else:
                #print('inputs: ', inputs)
                outputs, _ = net(inputs.unsqueeze(0)) 

            pred_py = F.softmax(outputs, dim=1).data
            del (outputs)
            if args.al_algorithm == 'bald':
                bald_map, pred_py = compute_bald(pred_py.cpu())
                predictions_py = pred_py.squeeze_(1).cpu().type(torch.FloatTensor)
                predictions_py_prob = None
            else:
                #continue
                #@carina uncomment class predictions
                pred_py = pred_py.max(1)
                predictions_py = pred_py[1].squeeze_(1).cpu().type(torch.FloatTensor)  #prediction of classes for each pixel
                predictions_py_prob = pred_py[0].squeeze_(1).cpu().type(torch.FloatTensor) #probability for each prediction
            if args.al_algorithm in ['entropy', 'ralis']:
                ent = compute_entropy_seg(args, inputs, net)
                #print('shape of entropy : ', ent.shape)

        if args.al_algorithm == 'bald':
            bald_region = bald_map[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                          int(candidates[1]):int(candidates[1]) + reg_sz[0]]
            bald_region = bald_region.mean()
            pred_region = predictions_py[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                          int(candidates[1]):int(candidates[1]) + reg_sz[0]]
            pred_region_prob = None
        else:
            # @carina 
            pred_region = predictions_py[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                          int(candidates[1]):int(candidates[1]) + reg_sz[0]]
            pred_region_prob = predictions_py_prob[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                               int(candidates[1]):int(candidates[1]) + reg_sz[0]]

        if args.al_algorithm == 'entropy':
            ent_region = ent[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],
                         int(candidates[1]):int(candidates[1]) + reg_sz[0]].mean()
            sample_stats = []
            sample_stats.append(ent_region.item())
        # Adding action representation
        elif args.al_algorithm == 'ralis':
            ent_region = ent[0, int(candidates[2]):int(candidates[2]) + reg_sz[1],  #ent.shape = torch.Size([1, 256, 256])
                        int(candidates[1]):int(candidates[1]) + reg_sz[0]]
            # in the third round, ent_region is tensor([], size=(0,80)) or with region size (32,32) it will be torch.Size([0,32])
            # Convert 2D maps into vector representation
            # if region size = 64x64 after the third ? round, torch.Size([0,64])
            #@carina uncomment sample stats
            #import ipdb 
            #ipdb.set_trace()
            sample_stats = create_feature_vector_3H_region_kl_sim_class_agnostic(ent_region, reg_sz=reg_sz)
        elif args.al_algorithm == 'random':
            sample_stats = []
            sample_stats.append(0.0)
        elif args.al_algorithm == 'bald':
            sample_stats = []
            sample_stats.append(bald_region.item())
        # unsqueeze returns a new tensor with a dimension of size one inserted at the specified position
        # tensor.detach() creates a tensor that shares storage with tensor that does not require grad. It detaches the output from the computational graph. 
        # So no gradient will be backpropagated along this variable.
        # The wrapper with torch.no_grad() temporarily set all the requires_grad flag to false. torch.no_grad says that no operation should build the graph
        state.append(torch.Tensor(sample_stats).unsqueeze(0).detach())

        del (pred_region)
        del (pred_region_prob)
        if args.al_algorithm in ['entropy', 'ralis']:
            del (ent_region)
        if args.al_algorithm == 'bald':
            del (bald_region)

    # end of for loop over candidates
    del (pred_py)
    del (predictions_py)
    del (predictions_py_prob)

    #if not cand == 256:
    state = torch.cat(state, dim=0)

    # Shuffle vector so that we do not end up with regions from the same image in the same pool
    # for candidates[2]==256: randperm=tensor([0, 3, 2, 1])
    randperm = torch.randperm(state.size()[0])
    state = state[randperm]
    region_candidates = np.array(region_candidates)[randperm]
    #print('region_candidates.shape: ', len(region_candidates))
    state = state.view(num_groups, state.size()[0] // num_groups, state.size()[1])
    #print('state.shape: ', state.size())
    region_candidates = np.reshape(region_candidates,
                                (num_groups, region_candidates.shape[0] // num_groups, region_candidates.shape[1]))
    state = Variable(state).cpu()  # [groups, cand_regions, channels, reg_size, reg_size]

    if args.al_algorithm == 'ralis':
        # Adding KL terms for action representation
        # @Carina uncomment kl_pool2
        #state = add_kl_pool2(state, n_cl=train_set.num_classes)
        # Add fixed part of state (from state subset)
        state_subset = []
        for index in range(0, len(candidate_set.state_subset)):
            inputs, gts, _, _, regions = candidate_set.get_subset_state(index) #inputs: torch.Size([3, 256, 256])
            inputs, gts = Variable(inputs).cuda(), Variable(gts).cuda()
            outputs, _ = net(inputs.unsqueeze(0)) #outputs = torch.Size([1, 4, 256, 256])
            pred_py = F.softmax(outputs).data # pred_py: torch.Size([1, 4, 256, 256])
            del (outputs)
            pred_py = pred_py.max(1) #.max(1) takes the max over dimension 1 and returns two tensors (max value in each row of pred_py, column index where max value is found).
            predictions_py = pred_py[1].squeeze_(1).cpu().type(torch.FloatTensor) #pred_py[1]=max values in each row; gives predictions for each pixel (torch.Size([1,256,256]))
            predictions_py_prob = pred_py[0].squeeze_(1).cpu().type(torch.FloatTensor) #gives probability for each prediction

            ent = compute_entropy_seg(args, inputs, net)  # ent.shape torch.Size([1, 256, 256])
            for reg in regions:
                pred_region_prob = predictions_py_prob[0, int(reg[1]):int(reg[1]) + reg_sz[1],
                                   int(reg[0]):int(reg[0]) + reg_sz[0]]
                pred_region = predictions_py[0, int(reg[1]):int(reg[1]) + reg_sz[1],
                              int(reg[0]):int(reg[0]) + reg_sz[0]]

                ent_region = ent[0, int(reg[1]):int(reg[1]) + reg_sz[1], int(reg[0]):int(reg[0]) + reg_sz[0]]
                # Convert 2D maps into vector representations
                # sample_stats = create_feature_vector_3H_region_kl(pred_region, ent_region,
                #                                                   num_classes=train_set.num_classes,
                #                                                   reg_sz=reg_sz)
                sample_stats = create_feature_vector_3H_region_kl_sim_class_agnostic(ent_region, reg_sz=reg_sz)
                state_subset.append(torch.Tensor(sample_stats).unsqueeze(0).detach())
                del (pred_region)
                del (pred_region_prob)
                del (ent_region)

            del (ent)
            del (pred_py)
            del (predictions_py)
            del (predictions_py_prob)
        state_subset = torch.cat(state_subset, dim=0)
    else:
        state_subset = None
    all_state = {'subset': state_subset, 'pool': state}
    if args.al_algorithm == 'bald':
        net.eval()
    print ('State computed! Time elapsed: ' + str(time.time() - s))
    #import ipdb
    #ipdb.set_trace()
    return all_state, region_candidates #len(all_state['subset']=288), len(region_candidates)=16

#@profile
def select_action(args, policy_net, all_state, steps_done, test=False):
    """We select the action: index of the image to label.
    :(argparse.ArgumentParser) args: The parser with all the defined arguments.
    :param policy_net: policy network.
    :param all_state: (torch.Variable) Torch tensor containing the state-action representation.
    :param steps_done: (int) Number of images labeled so far.
    :param test: (bool) Whether we are testing the DQN or training it.
    :return: Action (indexes of the regions to label), steps_done (updated counter), entropy in case of
    al_algorithm=='entropy' of the selected samples,
    """
    # import sys
    # from subprocess import call
    # # call(["nvcc", "--version"]) does not work
    # print('__Number CUDA Devices:', torch.cuda.device_count())
    # print('__Devices')
    # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    # print('Active CUDA Device: GPU', torch.cuda.current_device())
    # print ('Available devices ', torch.cuda.device_count())
    # print ('Current cuda device ', torch.cuda.current_device())

    state = all_state['pool']
    state_subset = all_state['subset']
    ent = 0
    if args.al_algorithm == 'ralis':
        policy_net.eval()
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        q_val_ = []
        if sample > eps_threshold or test:
            print ('Action selected with DQN!')
            with torch.no_grad():
                # Splitting state to fit it in GPU memory
                for i in range(0, state.size()[0], 16): #ipdb> state.size() torch.Size([3, 50, 262])  #state.shape [16, 8, 261],
                    print("state.shape: ", state.size())
                    if len(state_subset.shape) == 2: #state_subset.shape: torch.Size([288, 198])
                        print("state[i:i + 16].shape ", state[i:i + 16].shape) # train ralis 16,30,261 / [16,8,261];    for 16 being first dim:state[i:i+16] is same as state[:,:,:]
                        #print("policy_net(state[i:i + 16].cuda(), ", policy_net(state[i:i + 16].cuda(),  
                        #                          state_subset.unsqueeze(0).repeat(state[i:i + 16].size()[0], 1,
                        #                                                           1).cuda()))
                        print("state_subset_unsqeeze_dims: ", state_subset.unsqueeze(0).shape) #[1, 288,197] for brats18: [1,960,197]
                        print("state[i:i + 16].size()[0]: ", state[i:i + 16].size()[0])
                        # policy net takes as inputs: self, indexes_full_state=10 * 128, input_size=38, input_size_subset=38, sim_size=64
                        # unsqueeze(0) inserts dimension of size 1 at position 0
                        q_val_.append(policy_net(state[i:i + 16].cuda(),  
                                                 state_subset.unsqueeze(0).repeat(state[i:i + 16].size()[0], 1,    #repeats the tensor state_subset along the specified dimensions,
                                                                                  1).cuda()).cpu())                 # here dimensions: [16, 1, 1]
                    else:
                        q_val_.append(policy_net(state[i:i + 16].cuda(),
                                                 state_subset.unsqueeze(0).repeat(state[i:i + 16].size()[0], 1, 1,
                                                                                  1).cuda()).cpu())

                    state[i:i + 16].cpu()
                q_val_ = torch.cat(q_val_) 
                index = 1
                action = q_val_.max(index)[1].view(-1).cpu()
                del (q_val_)
                del (state)
            state_subset.cpu()
        else:
            action = Variable(torch.Tensor([np.random.choice(range(args.rl_pool), state.size()[0], replace=True)]).type(
                torch.LongTensor).view(-1))  # .cuda()  

    elif args.al_algorithm == 'random':
        action = np.random.choice(args.rl_pool, args.num_each_iter)
    elif args.al_algorithm == 'entropy':
        max_ent = torch.max(state[:, :, -1], dim=1)
        action = max_ent[1]  # Index where the maximum entropy is.
        ent = max_ent[0]
    elif args.al_algorithm == 'bald':
        max_ent = torch.max(state[:, :, -1], dim=1)
        action = max_ent[1]  # Index where the maximum BALD metric is.
        ent = max_ent[0]
    else:
        raise ValueError('Action type not recognized')

    return action, steps_done, ent # len(action) = 16


def select_action_dataset_agnostic(args, policy_net, all_state, steps_done, test=False):
    """We select the action: index of the image to label.
    :(argparse.ArgumentParser) args: The parser with all the defined arguments.
    :param policy_net: policy network.
    :param all_state: (torch.Variable) Torch tensor containing the state-action representation.
    :param steps_done: (int) Number of images labeled so far.
    :param test: (bool) Whether we are testing the DQN or training it.
    :return: Action (indexes of the regions to label), steps_done (updated counter), entropy in case of
    al_algorithm=='entropy' of the selected samples,
    """
    state = all_state['pool']
    state_subset = all_state['subset']
    ent = 0
    if args.al_algorithm == 'ralis':
        policy_net.eval()
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        q_val_ = []
        # import ipdb
        # ipdb.set_trace()
        if sample > eps_threshold or test:
            print ('Action selected with DQN!')
            with torch.no_grad():
                # Splitting state to fit it in GPU memory
                for i in range(0, state.size()[0], 16): #ipdb> state.size() torch.Size([3, 50, 262])

                    if len(state_subset.shape) == 2: #state_subset.shape: torch.Size([288, 198])
                        q_val_.append(policy_net(state[i:i + 16].cuda(),  #state[i:i + 16].shape = torch.Size([3, 50, 262])
                                                 state_subset.unsqueeze(0).repeat(state[i:i + 16].size()[0], 1,
                                                                                  1).cuda()).cpu())
                    else:
                        q_val_.append(policy_net(state[i:i + 16].cuda(),
                                                 state_subset.unsqueeze(0).repeat(state[i:i + 16].size()[0], 1, 1,
                                                                                  1).cuda()).cpu())

                    state[i:i + 16].cpu()
                q_val_ = torch.cat(q_val_)
                index = 1
                action = q_val_.max(index)[1].view(-1).cpu()
                del (q_val_)
                del (state)
            state_subset.cpu()
        else:
            action = Variable(torch.Tensor([np.random.choice(range(args.rl_pool), state.size()[0], replace=True)]).type(
                torch.LongTensor).view(-1))  # .cuda()

    elif args.al_algorithm == 'random':
        action = np.random.choice(args.rl_pool, args.num_each_iter)
    elif args.al_algorithm == 'entropy':
        max_ent = torch.max(state[:, :, -1], dim=1)
        action = max_ent[1]  # Index where the maximum entropy is.
        ent = max_ent[0]
    elif args.al_algorithm == 'bald':
        max_ent = torch.max(state[:, :, -1], dim=1)
        action = max_ent[1]  # Index where the maximum BALD metric is.
        ent = max_ent[0]
    else:
        raise ValueError('Action type not recognized')

    return action, steps_done, ent


def add_labeled_images(args, list_existing_images, region_candidates, train_set, action_list, budget, n_ep):
    """This function adds an image, indicated by 'action_list' out of 'region_candidates' list
     and adds it into the labeled dataset and the list of existing images.

    :(argparse.ArgumentParser) args: The parser with all the defined arguments.
    :param list_existing_images: (list) of tuples (image idx, region_x, region_y) of all regions that have
            been selected in the past to add them to the labeled set.
    :param region_candidates: (list) List of all possible regions to add.
    :param train_set: (torch.utils.data.Dataset) Training set.
    :param action_list: Selected indexes of the regions in 'region_candidates' to be labeled.
    :param budget: (int) Number of maximum regions we want to label.
    :param n_ep: (int) Number of episode.
    :return: List of existing images, updated with the new image.
    """

    lab_set = open(os.path.join(args.ckpt_path, args.exp_name, 'labeled_set_' + str(n_ep) + '.txt'), 'a')
    for i, action in enumerate(action_list):
        if train_set.get_num_labeled_regions() >= budget:
            print ('Budget reached with ' + str(train_set.get_num_labeled_regions()) + ' regions!')
            break
        im_toadd = region_candidates[i, action, 0]
        train_set.add_index(im_toadd, (region_candidates[i, action, 1], region_candidates[i, action, 2]))
        list_existing_images.append(tuple(region_candidates[i, action]))
        lab_set.write("%i,%i,%i" % (
            im_toadd, region_candidates[i, action, 1], region_candidates[i, action, 2]))
        lab_set.write("\n")
    print('Labeled set has now ' + str(train_set.get_num_labeled_regions()) + ' labeled regions.')

    return list_existing_images


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def compute_bald(predictions):
    ### Compute BALD ###
    expected_entropy = - torch.mean(torch.sum(predictions * torch.log(predictions + 1e-10), dim=1),
                                    dim=0)
    expected_p = torch.mean(predictions, dim=0)  # [batch_size, n_classes]
    pred_py = expected_p.max(0)[1]
    entropy_expected_p = - torch.sum(expected_p * torch.log(expected_p + 1e-10),
                                     dim=0)  # [batch size]
    bald_acq = entropy_expected_p - expected_entropy
    return bald_acq.unsqueeze(0), pred_py.unsqueeze(0)


def add_kl_pool2(state, n_cl=19):
    sim_matrix = torch.zeros((state.shape[0], state.shape[1], 32))
    all_cand = state[:, :, 0:n_cl + 1].view(-1, n_cl + 1).transpose(1, 0)
    for i in range(state.shape[0]):
        pool_hist = state[i, :, 0:n_cl + 1]
        for j in range(pool_hist.shape[0]):
            prov_sim = entropy(pool_hist[j:j + 1].repeat(all_cand.shape[1], 1).transpose(0, 1), all_cand)
            hist, _ = np.histogram(prov_sim, bins=32)
            hist = hist / hist.sum()
            sim_matrix[i, j, :] = torch.Tensor(hist)
    state = torch.cat([state.cuda(), sim_matrix.cuda()], dim=2) #@carina added .cuda()
    return state

def create_feature_vector_3H_region_kl_sim(pred_region, ent_region, train_set, num_classes=19, reg_sz=(128, 128)):
    unique, counts = np.unique(pred_region, return_counts=True)
    sample_stats = np.zeros(num_classes + 1) + 1E-7
    sample_stats[unique.astype(int)] = counts
    sample_stats = sample_stats.tolist()
    sz = ent_region.size()
    ks_x = int(reg_sz[0] // 8)
    ks_y = int(reg_sz[1] // 8)
    with torch.no_grad():
        sample_stats += (5 - F.max_pool2d(5 - ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(
            -1)).tolist()  # min entropy  
        sample_stats += F.avg_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
        #print("avg_pool: ",  "ent_region", ent_region.shape, "len of sample_stats: ", len(sample_stats), "ks_x: ", ks_x, "ks_y: ", ks_y)

        #print('Second F.max_pool2d.shape: ', F.max_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).shape)
        sample_stats += F.max_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
        #print("second max_pool: ", "ent_region", ent_region.shape, "len of sample_stats: ", len(sample_stats), "ks_x: ", ks_x, "ks_y: ", ks_y)
    if len(train_set.balance_cl) > 0:
        inp_hist = sample_stats[0:num_classes + 1]
        sim_sample = entropy(np.repeat(np.asarray(inp_hist)[:, np.newaxis], len(train_set.balance_cl), axis=1),
                             np.asarray(train_set.balance_cl).transpose(1, 0))
        hist, _ = np.histogram(sim_sample, bins=32)
        sim_lab = list(hist / hist.sum())
        sample_stats += sim_lab
    else:
        # add 32 zeros to sample_stats - why?
        # @carina uncomment the following line
        sample_stats += [0.0] * 32
    #print("return sample_stats: ", "len of sample_stats: ", len(sample_stats))
    return sample_stats

def create_feature_vector_3H_region_kl_sim_class_agnostic(ent_region, reg_sz=(128, 128)):
    #unique, counts = np.unique(pred_region, return_counts=True) #unique is [1.,2.,3.] classes present, counts how many of each class are present [3880, 139, 77]
    #sample_stats = np.zeros(num_classes + 1) + 1E-7 @carina
    sample_stats = []
    #np.array(sample_stats)[unique.astype(int)] = counts
    #sample_stats = sample_stats.tolist()
    sz = ent_region.size()
    ks_x = int(reg_sz[0] // 8)
    ks_y = int(reg_sz[1] // 8)
    with torch.no_grad():
        #import ipdb
        #ipdb.set_trace()
        #print("first max_pool: ", "ent_region", ent_region.shape, "ks_x: ", ks_x, "ks_y: ", ks_y)
        #print('F.max_pool2d.shape: ', F.max_pool2d(5 - ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).shape)
        sample_stats += (5 - F.max_pool2d(5 - ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(
            -1)).tolist()  # min entropy  
        #print('F.avg_pool2d: ', F.avg_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)))
        #print('F.avg_pool2d.shape: ', F.avg_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).shape)
        sample_stats += F.avg_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
        #print("avg_pool: ",  "ent_region", ent_region.shape, "len of sample_stats: ", len(sample_stats), "ks_x: ", ks_x, "ks_y: ", ks_y)

        #print('Second F.max_pool2d.shape: ', F.max_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).shape)
        sample_stats += F.max_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
        #print("second max_pool: ", "ent_region", ent_region.shape, "len of sample_stats: ", len(sample_stats), "ks_x: ", ks_x, "ks_y: ", ks_y)
    # if len(train_set.balance_cl) > 0:
    #     #inp_hist = sample_stats[0:num_classes + 1]
    #     sim_sample = entropy(np.repeat(np.asarray(inp_hist)[:, np.newaxis], len(train_set.balance_cl), axis=1),
    #                          np.asarray(train_set.balance_cl).transpose(1, 0))
    #     hist, _ = np.histogram(sim_sample, bins=32)
    #     sim_lab = list(hist / hist.sum())
    #     sample_stats += sim_lab
    # else:
    #     # add 32 zeros to sample_stats - why?
    #     # @carina uncomment the following line
    #sample_stats += [0.0] * 32
    #print("len of sample_stats: ", len(sample_stats))
    return sample_stats #len(sample_stats) is 192


def create_feature_vector_3H_region_kl(pred_region, ent_region, num_classes=19, reg_sz=(128, 128)):
    #import ipdb
    #ipdb.set_trace()
    unique, counts = np.unique(pred_region, return_counts=True)
    sample_stats = np.zeros(num_classes + 1) + 1E-7
    sample_stats[unique.astype(int)] = counts
    sample_stats = sample_stats.tolist()
    sz = ent_region.size()
    ks_x = int(reg_sz[0] // 8)
    ks_y = int(reg_sz[1] // 8)
    with torch.no_grad():
        sample_stats += (5 - F.max_pool2d(5 - ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(
            -1)).tolist()  # min entropy
        sample_stats += F.avg_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
        sample_stats += F.max_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
    return sample_stats


def compute_entropy_seg(args, im_t, net):
    '''
    Compute entropy function
    :param args:
    :param im_t:
    :param net:
    :return:
    '''
    net.eval()
    if im_t.dim() == 3:
        im_t_sz = im_t.size()
        im_t = im_t.view(1, im_t_sz[0], im_t_sz[1], im_t_sz[2])

    out, _ = net(im_t)
    out_soft_log = F.log_softmax(out) 
    out_soft = torch.exp(out_soft_log) #max in softmax: exponential of softmax, increases the probability of the biggest score and decreases the probability of the lower score
    ent = - torch.sum(out_soft * out_soft_log, dim=1).detach().cpu()  # .data.numpy()
    del (out)
    del (out_soft_log)
    del (out_soft)
    del (im_t)

    return ent

#@profile
def optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP, BATCH_SIZE=32, GAMMA=0.999,
                        dqn_epochs=1):
    """This function optimizes the policy network

    :(ReplayMemory) memory: Experience replay buffer
    :param Transition: definition of the experience replay tuple
    :param policy_net: Policy network
    :param target_net: Target network
    :param optimizerP: Optimizer of the policy network
    :param BATCH_SIZE: (int) Batch size to sample from the experience replay
    :param GAMMA: (float) Discount factor
    :param dqn_epochs: (int) Number of epochs to train the DQN
    """
    # Code adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    if len(memory) < BATCH_SIZE:
        return
    print('Optimize model...')
    print ('memory: ', len(memory))
    policy_net.train()
    loss_item = 0
    # import ipdb
    # ipdb.set_trace()
    for ep in range(dqn_epochs):
        optimizerP.zero_grad()
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool).cuda() #@carina changed torch.uint8 to torch.bool

        non_final_next_states = torch.cat([s.cuda() for s in batch.next_state #@carina added cuda
                                           if s is not None])
        non_final_next_states_subset = torch.cat([s.cuda() for s in batch.next_state_subset #@carina added cuda
                                                  if s is not None])

        state_batch = torch.cat(batch.state) #torch.cat concatenates given tensors in dim=0 (default), torch.Size([16, 30, 192]) 
        state_batch_subset = torch.cat(batch.state_subset) 
        action_batch = torch.Tensor([batch.action]).view(-1).type(torch.LongTensor)
        reward_batch = torch.Tensor([batch.reward]).view(-1).cuda()
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_val = policy_net(state_batch.cuda(), state_batch_subset.cuda()) # state_batch.shape = [16,8,261], state_batch_subset = [16,288,197]
        state_batch.cpu()
        state_batch_subset.cpu()
        #print("q_val.shape: ", q_val.shape) #[16, 8] for pool-size = 8 
        #print("action_batch unsqueeze.shape", action_batch.unsqueeze(1).shape) # [16,1]
        state_action_values = q_val.gather(1, action_batch.unsqueeze(1).cuda()) #[16,1]
        #print("state_action_values.shape: ", state_action_values.shape)
        action_batch.cpu()
        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE).cuda()
        # Double dqn, so we compute the values with the target network, we choose the actions with the policy network
        if non_final_mask.sum().item() > 0:
            print("-------------------------------------- on_final_mask.sum().item() > 0 --------------------------")
            v_val_act = policy_net(non_final_next_states.cuda(), non_final_next_states_subset.cuda()).detach()
            v_val = target_net(non_final_next_states.cuda(), non_final_next_states_subset.cuda()).detach()
            non_final_next_states.cpu()
            non_final_next_states_subset.cpu()
            act = v_val_act.max(1)[1]
            next_state_values[non_final_mask] = v_val.gather(1, act.unsqueeze(1)).view(-1).cuda()  
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.view(-1), expected_state_action_values)
        loss_item += loss.item()
        progress_bar(ep, dqn_epochs, '[DQN loss %.5f]' % (
                loss_item / (ep + 1)))
        loss.backward()
        optimizerP.step()

        del (q_val)
        del (expected_state_action_values)
        del (loss)
        del (next_state_values)
        del (reward_batch)
        if non_final_mask.sum().item() > 0:
            del (act)
            del (v_val)
            del (v_val_act)
        del (state_action_values)
        del (state_batch)
        del (action_batch)
        del (non_final_mask)
        del (non_final_next_states)
        del (batch)
        del (transitions)
    lab_set = open(os.path.join(args.ckpt_path, args.exp_name, 'q_loss.txt'), 'a')
    lab_set.write("%f" % (loss_item))
    lab_set.write("\n")
    lab_set.close()
