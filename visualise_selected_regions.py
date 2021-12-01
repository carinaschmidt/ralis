# Pseudo code for predicting a single image
#net, _, _ = create_models(**kwargs_models)
#net.eval()
#net_dict = torch.load(net_checkpoint_path)
#net.load_state_dict(net_dict)
#img = myload(path)
#img = my_to_tensor(img)
#output, _ = net(img)
#output = maybe_my_argmax(output)
from models.model_utils import create_models, load_models, get_region_candidates, compute_state, select_action, \
    add_labeled_images, optimize_model_conv
from data.data_utils import get_data
from data import acdc_al
import torch
import os
import utils.parser as parser
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
#ckpt_path = '/home/baumgartner/cschmidt77/ckpt_seg_acdc'
#ckpt_path = '/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_acdc'
#data_path = '/mnt/qb/baumgartner/cschmidt77_data'
#code_path = '/home/baumgartner/cschmidt77/devel/ralis'
#run locally
#code_path = '/home/carina/baumgartner/cschmidt77/devel/ralis'

# ckpt_path='/home/carina/Desktop/2021-07-27/mnt_qb_baumgartner_cschmidt77_data/ckpt_seg_acdc'
# data_path='/home/carina/Desktop/2021-07-27/mnt_qb_baumgartner_cschmidt77_data'
# code_path ='/home/carina/Desktop/ralis-new-master'
# sif_path='/home/carina/tue-slurm-helloworld/ralis.sif'

def main(args):
    ####------ Create segmentation, query and target networks ------####
    kwargs_models = {"dataset": args.dataset,
                "al_algorithm": args.al_algorithm,
                "region_size": args.region_size
            }

    #segmentation net, policy net
    net, policy_net, target_net = create_models(**kwargs_models)
    net.eval()

    net_checkpoint_path = os.path.join(args.ckpt_path, args.exp_name_toload, 'best_jaccard_val.pth') #'/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_new/pretraining_acdc_06-19/last_jaccard_val.pth'
    net_dict = torch.load(net_checkpoint_path)
    net.load_state_dict(net_dict)
    root = os.path.join(args.data_path, args.dataset) 

    policy_path = os.path.join(args.ckpt_path, args.exp_name_toload_rl, 'policy_' + args.snapshot)

    ####------ Load policy (RL) from one folder and network from another folder ------####
    policy_net.load_state_dict(torch.load(policy_path))


    num_regions = args.num_each_iter * args.rl_pool
    num_groups = args.num_each_iter

    train_set = acdc_al.ACDC_al('fine', 'train',
                                            data_path=args.data_path,
                                            code_path=args.code_path,
                                            joint_transform=None,
                                            transform=None,
                                            target_transform=None, num_each_iter=args.num_each_iter,
                                            only_last_labeled=None,
                                            split='train' if args.al_algorithm == 'ralis' and not args.test else 'test', #if --train and --test, still test=TRUE
                                            region_size=args.region_size)

    import ipdb
    ipdb.set_trace()
    # mode is 'train', split is 'test' (default) -> d_v 
    # candidate_set: image indexes to be potentially labeled @carina 
    train_loader = DataLoader(train_set,
                            batch_size=args.train_batch_size,
                            num_workers=0, shuffle=True,
                            drop_last=False)

    candidate_set = acdc_al.ACDC_al('fine', 'train',
                                    data_path=args.data_path,
                                    code_path=args.code_path,
                                    joint_transform=None,
                                    candidates_option=True,
                                    transform=None,
                                    target_transform=None,
                                    split='train' if args.al_algorithm == 'ralis' and not args.test else 'test',
                                    region_size=args.region_size)

    # _, train_set, _, candidate_set = get_data(**kwargs_data)
    # train_loader, train_set, val_loader, candidate_set

    # Get candidates for state , num_regions = num.each_iter*args.rl_pool
    candidates = train_set.get_candidates(num_regions_unlab=num_regions) #take all regions from image into account
    candidate_set.reset()
    candidate_set.add_index(list(candidates)) 

    # Choose candidate pool, filtering out the images we already have
    # List of tuples (int(Image index), int(width_coord), int(height_coord))
    region_candidates = get_region_candidates(candidates, train_set, num_regions=num_regions) #region candidates = state_set D_s? @carina
    
    current_state, region_candidates = compute_state(args, net, region_candidates, candidate_set, train_set,
                                                             num_groups=num_groups, reg_sz=args.region_size)
    action, steps_done, chosen_stats = select_action(args, policy_net, current_state,
                                                                 0, test=True)
    list_existing_images = []
    list_existing_images = add_labeled_images(args, list_existing_images=list_existing_images,
                                                          region_candidates=region_candidates, train_set=train_set,
                                                          action_list=action, budget=args.budget_labels, n_ep=1) #n_ep in range num(episodes) which is rl_episodes
    #create plot for every candidate region
    # get image by id, select region from image and plot it  

    for cr in region_candidates:
        train_set.get_specific_item(cr[0])
        # cr[0] is first tuple
        plt.imshow(cr)

    # for i, action in enumerate(action_list):
    #     if train_set.get_num_labeled_regions() >= budget:
    #         print ('Budget reached with ' + str(train_set.get_num_labeled_regions()) + ' regions!')
    #         break
    #     im_toadd = region_candidates[i, action, 0]
    #     train_set.add_index(im_toadd, (region_candidates[i, action, 1], region_candidates[i, action, 2]))
    #     list_existing_images.append(tuple(region_candidates[i, action]))
    #     lab_set.write("%i,%i,%i" % (
    #         im_toadd, region_candidates[i, action, 1], region_candidates[i, action, 2]))
    #     lab_set.write("\n")
    # print('Labeled set has now ' + str(train_set.get_num_labeled_regions()) + ' labeled regions.')
    # img_path, mask_path, im_name = self.imgs[img]
    # #mask = Image.open(mask_path)
    # mask = np.load(mask_path)
    # #mask = np.array(mask)
    # r_x = int(region[1])
    # r_y = int(region[0])
    # region_classes = mask[r_x: r_x + region_size[1], r_y: r_y + region_size[0]]

    # for img_name in img_names:
    #     img_path = os.path.join(root, "slices", "train", img_name)
    #     mask_path = os.path.join(root, "gt", "train", img_name)
    #     img_np, mask_np = np.load(img_path), np.load(mask_path)
    #     img, mask = torch.from_numpy(img_np), torch.from_numpy(mask_np)
    #     img = torch.stack((img, img, img), dim=0)

    # #load a few images and masks
    # img_names = ['pat_4_diag_2_frame_01_slice_0.npy']

    # for img_name in img_names:
    #     img_path = os.path.join(root, "slices", "train", img_name)
    #     mask_path = os.path.join(root, "gt", "train", img_name)
    #     img_np, mask_np = np.load(img_path), np.load(mask_path)
    #     img, mask = torch.from_numpy(img_np), torch.from_numpy(mask_np)
    #     img = torch.stack((img, img, img), dim=0)

    #     im_t = img
    #     if im_t.dim() == 3:
    #         im_t_sz = im_t.size()
    #         im_t = im_t.view(1, im_t_sz[0], im_t_sz[1], im_t_sz[2])
    #         im_t = Variable(im_t).cuda()

    #     output, _ = net(im_t)
    #     # Get segmentation maps
    #     predictions_py = output.data.max(1)[1].squeeze_(1)

    #     pred_cpu = predictions_py.cpu()
    #     preds = pred_cpu.detach().numpy()

    #     im_t = im_t.cpu()
    #     imt = im_t.detach().numpy()

    #     #plt.figure()
    #     fig, ax = plt.subplots(1,3)
    #     ax[0].imshow(imt[-1,-1,:,:],cmap='gray') #image
    #     ax[0].set_title("MRI slice")
    #     ax[1].imshow(mask) #mask
    #     ax[1].set_title("Mask")
    #     ax[2].imshow(preds[-1,:,:]) #prediction
    #     ax[2].set_title("Prediction")
    #     #plt.show()
    #     plt.savefig(os.path.join('/home/carina/Desktop/show_images/' +'2021-07-27-regions' + img_name.split('.')[0] + '.png'))


if __name__ == '__main__':
    ####------ Parse arguments from console  ------####
    args = parser.get_arguments()
    main(args)