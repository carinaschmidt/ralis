# Pseudo code for visualising regions
# load policy net, load labeled set list 
#net, _, _ = create_models(**kwargs_models)
#net.eval()
#net_dict = torch.load(net_checkpoint_path)
#net.load_state_dict(net_dict)
#img = myload(path)
#img = my_to_tensor(img)
#output, _ = net(img)
#output = maybe_my_argmax(output)
from torch.utils import data
from models.model_utils import create_models, load_models, get_region_candidates, compute_state, select_action, \
    add_labeled_images, optimize_model_conv
from data.data_utils import get_data
from data import acdc_al, brats18_2D_al
import torch
import os
import utils.parser as parser
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Rectangle
from matplotlib import colors
import matplotlib.cm as cm

#ckpt_path = '/home/baumgartner/cschmidt77/ckpt_seg_acdc'
#ckpt_path = '/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_acdc'
#data_path = '/mnt/qb/baumgartner/cschmidt77_data'
#code_path = '/home/baumgartner/cschmidt77/devel/ralis'
#run locally
#code_path = '/home/carina/baumgartner/cschmidt77/devel/ralis'


def main(args):
    dataset = args.dataset
    ####------ Create segmentation, query and target networks ------####

    # create colormap for matplotlib
    #plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#000000","#FFFF00","#FF0000","#10A5F5"]) 
    plt.rcParams['font.size'] = '8'
    # plt.rc('text', usetex=True)
    # plt.rc('font', **{'sans-serif': ['lmodern'], 'size': 11})
    # make a color map of fixed colors
    myColors = colors.ListedColormap(['black', 'blue', 'gold', 'magenta'])

    if args.dataset == 'acdc':
        train_set = acdc_al.ACDC_al('fine', 'train',
                                                data_path=args.data_path,
                                                code_path=args.code_path,
                                                joint_transform=None,
                                                transform=None,
                                                target_transform=None, num_each_iter=args.num_each_iter,
                                                only_last_labeled=None,
                                                split='train' if args.al_algorithm == 'ralis' and not args.test else 'test', #if --train and --test, still test=TRUE
                                                region_size=args.region_size)
    elif args.dataset == 'brats18':
        myColors = colors.ListedColormap(['black', 'darkred', 'orange', 'antiquewhite'])
        print("al algo: ", args.al_algorithm)
        train_set = brats18_2D_al.BraTS18_2D_al('fine', 'train',
                                data_path=args.data_path,
                                code_path=args.code_path,
                                joint_transform=None,
                                transform=None,
                                target_transform=None, num_each_iter=args.num_each_iter,
                                only_last_labeled=args.only_last_labeled,
                                split='train' if args.al_algorithm == 'ralis' and not args.test else 'test',
                                region_size=args.region_size)

    list_images = train_set.imgs
    n_ep = 0
    # dict of image indexes already labelled as key, coordinates as value
    idx_region_dict = {}

    #labeled set file
    
    if args.dataset == 'acdc':
        ralis_path = open(os.path.join(args.ckpt_path, args.exp_name_toload, 'labeled_set_' + str(n_ep) + '.txt'), 'r')
        entropy_path = open(os.path.join('/mnt/qb/baumgartner/cschmidt77_data/exp2_acdc_baselines_preTrainDT/2021-11-03-acdc_ImageNetBackbone_baseline_entropy_budget_2384_seed_123/', 'labeled_set_0.txt'), 'r')
        bald_path = open(os.path.join('/mnt/qb/baumgartner/cschmidt77_data/exp2_acdc_baselines_preTrainDT/2021-11-04-acdc_ImageNetBackbone_baseline_bald_budget_2384_seed_123/', 'labeled_set_0.txt'), 'r')
        random_path = open(os.path.join('/mnt/qb/baumgartner/cschmidt77_data/exp2_acdc_baselines_preTrainDT/2021-11-03-acdc_ImageNetBackbone_baseline_random_budget_2384_seed_123/', 'labeled_set_' + str(n_ep) + '.txt'), 'r')
    elif args.dataset == 'brats18': # 'brats18'
        n_ep = 36
        ralis_path = open(os.path.join(args.ckpt_path, args.exp_name_toload, 'labeled_set_' + str(n_ep) + '.txt'), 'r') #ckpt_path: /mnt/qb/baumgartner/cschmidt77_data/exp1b_brats_baselines/2021-11-04-brats18_ImageNetBackbone_baseline_random_budget_17792_seed_123
        entropy_path = open(os.path.join('/mnt/qb/baumgartner/cschmidt77_data/exp1b_brats_baselines/2021-11-04-brats18_ImageNetBackbone_baseline_entropy_budget_17792_seed_123/', 'labeled_set_0.txt'), 'r')
        bald_path = open(os.path.join('/mnt/qb/baumgartner/cschmidt77_data/exp1b_brats_baselines/2021-11-04-brats18_ImageNetBackbone_baseline_bald_budget_17792_seed_234', 'labeled_set_0.txt'), 'r')
        random_path = open(os.path.join('/mnt/qb/baumgartner/cschmidt77_data/exp1b_brats_baselines/2021-11-04-brats18_ImageNetBackbone_baseline_random_budget_17792_seed_123/', 'labeled_set_0.txt'), 'r')

    print("ralis_path: ", ralis_path)
    file_paths = [ralis_path, entropy_path, bald_path, random_path]

# for image: save intensity min and intensity max

# for image:
# plt.imshow(x, cmin=1.1, cmax=0.9 von imax)

    for i, alalgo in enumerate(file_paths):
        print("alalgo: ", alalgo)
        file_path = alalgo
        print("i: ", i)
        if i == 0:
            al = 'ralis'
        elif i == 1:
            al = 'entropy'
        elif i == 2:
            al = 'bald'
        elif i == 3:
            al = 'random'
        else:
            print("AL algo not recognised")

        for line in file_path:
            #line = file.readline()
            # get img indices and region coordinates from labelled set
            img_idx, coord_x_left_upper, coord_y_left_upper = line.rstrip('\n').split(',') #removes \n and splits by ,
            img_idx, coord_x_left_upper, coord_y_left_upper = int(img_idx), int(coord_x_left_upper), int(coord_y_left_upper)

            # add selected img idx to dict
            # if idx already added, append with new region coordinates, else create new key
            if img_idx in idx_region_dict.keys():
                idx_region_dict[img_idx].append((coord_x_left_upper, coord_y_left_upper))
            else:
                idx_region_dict[img_idx] = [(coord_x_left_upper, coord_y_left_upper)]

        v_minimum = 0.5
        v_maximum = 0.5
        i = 0
        # get intesity values 
        for key, values in idx_region_dict.items():
            img_idx = key
            img_path, mask_path, img_name = list_images[img_idx]
            print("img_path: ", img_path)
            img, mask = np.load(img_path), np.load(mask_path)
            vmin, vmax = img.min(), img.max()
            if vmin < v_minimum:
                v_minimum = vmin
            if vmax > v_maximum:
                v_maximum = vmax
            i +=1
            if i == 80:
                break

        print("vminimum: ", v_minimum)
        print("vmaximum: ", v_maximum)
        # iterate over dictionary with key: idx image, values: list of coordinate pairs  
        i = 0
        for key, values in idx_region_dict.items():
            img_idx = key

            img_path, mask_path, img_name = list_images[img_idx]
            print("img_path: ", img_path)
            img, mask = np.load(img_path), np.load(mask_path)
            if args.dataset == 'brats18':
                img = img[:,:,1]
            print("shape of img: ", img.shape)
            coordinate_pairs = values
            #region_img = img
            #region_mask = mask

            masked = np.full(mask.shape, 0)
            image_masked = img
            fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,7)) #,ax3,ax4)
            
            #ax1.imshow(img, cmap=cm.Greys_r, vmin=v_minimum, vmax=v_maximum) 
            ax1.imshow(img, cmap=cm.Greys_r, vmin=0.9*v_minimum, vmax=0.9*v_maximum) #mask
            ax1.set_title("MRI slice with selected regions")
            ax1.axis('off')

            ax2.imshow(mask, cmap=myColors) #mask
            ax2.set_title("Segmentation with selected regions")
            ax2.axis('off')

            cropped_regions = []
            cropped_masks = []
            # for each region coordinates
            names_list = []

            for pair in coordinate_pairs:
                coord_x_left_upper = pair[1]
                coord_y_left_upper = pair[0]

                #print("coord_left_upper (x,y): ", (coord_x_left_upper, coord_y_left_upper))

                coord_x_right_bottom = coord_x_left_upper + args.region_size[1] #region size is here 64, 48
                coord_y_right_bottom = coord_y_left_upper + args.region_size[0] #region size 40 or 48

                #print("coord right bottom (x,y): ", (coord_x_right_bottom, coord_y_right_bottom))

                # crop regions
                region_img = img[coord_y_left_upper: coord_y_right_bottom, coord_x_left_upper: coord_x_right_bottom]
                region_mask = mask[coord_y_left_upper: coord_y_right_bottom, coord_x_left_upper: coord_x_right_bottom]

                cropped_regions.append(region_img)
                cropped_masks.append(region_mask)

                # mask out region of image
                masked[coord_y_left_upper: coord_y_right_bottom, coord_x_left_upper: coord_x_right_bottom] = region_mask
                image_masked[coord_y_left_upper: coord_y_right_bottom, coord_x_left_upper: coord_x_right_bottom] = region_img

                # Create a Rectangle patch
                #print("coord_x_left_upper, y_left_upper: ", (coord_x_left_upper, coord_y_left_upper))
                #print("coord_x_right_bottom, y_right_bottom: ", (coord_x_right_bottom, coord_y_right_bottom))
                coord_x_left_lower =coord_x_left_upper 
                coord_y_left_lower =coord_y_left_upper + args.region_size[0]
                #print("coord_x_left_lower, y_left_lower: ", (coord_x_left_lower, coord_y_left_lower))
                # (x_low, y_low), width, height (48,40)
                rect = Rectangle((coord_x_left_lower, coord_y_left_lower),args.region_size[1],args.region_size[0],linewidth=0.8,edgecolor='lime',facecolor='none')
                # Add the patch to the Axes
                ax1.add_patch(rect)
                rect = Rectangle((coord_x_left_lower,coord_y_left_lower),args.region_size[1],args.region_size[0],linewidth=0.8,edgecolor='lime',facecolor='none')
                ax2.add_patch(rect)
                #img_copy[coord_y_left_upper: coord_y_right_bottom, coord_x_left_upper: coord_x_right_bottom] = 0  
                #name = os.path.join('/home/carina/Desktop/show_regions/', al, img_name)
                name = os.path.join('/home/carina/Desktop/regions_visualisation_RALIS_DQN/brats18-dqn', al, str(i) + "_" + img_name) #str(i))#
                #name = str(i) + "_" + name 
                print("name: ", name)
                plt.savefig(f'{name}.png', bbox_inches='tight')
                #plt.close()
                #plt.savefig(f'{name}.png')#, bbox_inches='tight')
                i += 1
    alalgo.close()


def rc_params():
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'serif', 'sans-serif': ['lmodern'], 'size': 20})
    plt.rc('axes', **{'titlesize': 22, 'labelsize': 22})
    plt.rc('xtick', **{'labelsize': 18})
    plt.rc('ytick', **{'labelsize': 18})
    plt.rc('legend', **{'fontsize': 18})
    plt.rc('figure', **{'figsize': (12,7)})

# Experiment to load for presentation:
# /mnt/qb/baumgartner/cschmidt77_data/exp4_acdc_train_DT_small/2021-10-26-train_acdc_ImageNetBackbone_budget_128_lr_0.05_2patients_seed123
# labeled_set_0.txt
# labeled_set_49.txt


# #/mnt/qb/baumgartner/cschmidt77_data/ACDC_regionsize_3232/2021-11-19-acdc_3232_train_2patients_ImageNetBackbone_budget_512_lr_0.05_seed_123
# singularity exec --nv --bind '/mnt/qb/baumgartner/cschmidt77_data/' '/home/carina/tue-slurm-helloworld/ralis.sif' python3 -u '/home/carina/ralis/visualise_regions.py' 
# --exp-name-toload '2021-11-19-acdc_3232_train_2patients_ImageNetBackbone_budget_512_lr_0.05_seed_123' --checkpointer --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/ACDC_regionsize_3232/'  
# --data-path '/mnt/qb/baumgartner/cschmidt77_data/' --input-size 128 128 --dataset 'acdc' --al-algorithm 'ralis' --region-size 32 32  
# --train-batch-size 2 --val-batch-size 1 --exp-name-toload-rl '2021-07-17-train_acdc_ImageNetBackbone_budget_608_lr_0.01_seed_123' --num-each-iter 1 --rl-pool 30 --test


# trained DQN:
#/mnt/qb/baumgartner/cschmidt77_data/exp1b_brats_baselines/2021-10-31-brats18_ImageNetBackbone_stdAug_budget_1536_lr_0.01_seed_55

# brats
# singularity exec --nv --bind '/mnt/qb/baumgartner/cschmidt77_data/' '/home/carina/tue-slurm-helloworld/ralis.sif' python3 -u '/home/carina/ralis/visualise_regions.py' --exp-name-toload '2021-10-31-brats18_ImageNetBackbone_stdAug_budget_1536_lr_0.01_seed_55' --checkpointer --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/exp1b_brats_baselines/' --data-path '/mnt/qb/baumgartner/cschmidt77_data/' --input-size 128 128 --dataset 'brats18' --al-algorithm 'ralis' --region-size 40 48 --train --test --final-test


if __name__ == '__main__':
    ####------ Parse arguments from console  ------####
    args = parser.get_arguments()
    main(args)