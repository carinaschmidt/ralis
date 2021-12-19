# Pseudo code for predicting a single image
#net, _, _ = create_models(**kwargs_models)
#net.eval()
#net_dict = torch.load(net_checkpoint_path)
#net.load_state_dict(net_dict)
#img = myload(path)
#img = my_to_tensor(img)
#output, _ = net(img)
#output = maybe_my_argmax(output)
from models.model_utils import create_models
import torch
import os
import utils.parser as parser
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from skimage import data, segmentation
# ckpt_path = '/home/baumgartner/cschmidt77/ckpt_seg'
data_path = '/mnt/qb/baumgartner/cschmidt77_data'
#code_path = '/home/baumgartner/cschmidt77/devel/ralis'
#run locally
code_path = '/home/carina/baumgartner/cschmidt77/devel/ralis'

# visualise MSD Heart image data
def main(args):
    ####------ Create segmentation, query and target networks ------####
    kwargs_models = {"dataset": args.dataset,
                "al_algorithm": args.al_algorithm,
                "region_size": args.region_size
            }

    #segmentation net
    net, _, _ = create_models(**kwargs_models)
    net.eval()

    # Experiment name to load weights from: exp_name_toload = 'pretraining_acdc_06-19'
    # Path to store weights, logs and other experiment related files

    net_checkpoint_path = os.path.join(args.ckpt_path, args.exp_name_toload, 'best_jaccard_val.pth') #'/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_new/pretraining_acdc_06-19/last_jaccard_val.pth'
    #net_path = os.path.join(ckpt_path, exp_name_toload, 'best_jaccard_val.pth')
    net_dict = torch.load(net_checkpoint_path)
    net.load_state_dict(net_dict)
    root = os.path.join(args.data_path, 'msd_heart') 

    #load a few images and masks
    img_names = ['la_pat_011_slice_17.npy', 'la_pat_021_slice_12.npy','la_pat_030_slice_83.npy', 'la_pat_011_slice_18.npy',   'la_pat_021_slice_13.npy', 'la_pat_030_slice_84.npy',
'la_pat_011_slice_19.npy',   'la_pat_021_slice_14.npy',   'la_pat_030_slice_85.npy',
'la_pat_011_slice_1.npy',    'la_pat_021_slice_15.npy',   'la_pat_030_slice_86.npy',
'la_pat_011_slice_20.npy',   'la_pat_021_slice_16.npy',   'la_pat_030_slice_87.npy',
'la_pat_011_slice_21.npy',   'la_pat_021_slice_17.npy',   'la_pat_030_slice_88.npy',
'la_pat_011_slice_22.npy',   'la_pat_021_slice_18.npy',   'la_pat_030_slice_89.npy',
'la_pat_011_slice_23.npy',   'la_pat_021_slice_19.npy',   'la_pat_030_slice_8.npy',
'la_pat_011_slice_24.npy',   'la_pat_021_slice_1.npy',    'la_pat_030_slice_90.npy',
'la_pat_011_slice_25.npy',   'la_pat_021_slice_20.npy',   'la_pat_030_slice_91.npy']

    for img_name in img_names:
        img_path = os.path.join(root, "slices", img_name)
        mask_path = os.path.join(root, "gt", img_name)
        img_np, mask_np = np.load(img_path), np.load(mask_path)
        img, mask = torch.from_numpy(img_np), torch.from_numpy(mask_np)
        img = torch.stack((img, img, img), dim=0)

        im_t = img
        if im_t.dim() == 3:
            im_t_sz = im_t.size()
            im_t = im_t.view(1, im_t_sz[0], im_t_sz[1], im_t_sz[2])
            im_t = Variable(im_t).cuda()

        output, _ = net(im_t)
        # Get segmentation maps
        predictions_py = output.data.max(1)[1].squeeze_(1)

        pred_cpu = predictions_py.cpu()
        preds = pred_cpu.detach().numpy()

        im_t = im_t.cpu()
        imt = im_t.detach().numpy()

        #plt.figure()
        fig, ax = plt.subplots(1,3, figsize=(10,5))

        ax[0].imshow(imt[-1,-1,:,:],cmap='gray', interpolation='nearest') #image
        ax[0].set_title("MRI slice")
        ax[1].imshow(mask, interpolation='none') #mask
        #ax[1].imshow(clean_border, interpolation='nearest')
        ax[1].set_title("Mask")
        ax[2].imshow(preds[-1,:,:], interpolation='none') #prediction
        ax[2].set_title("Prediction")
        #plt.show()
        plt.savefig(os.path.join('/home/carina/Desktop/show_images/' +'2021-07-19_img_msdHeart_gt_pred_lr05_interpolationNone' + img_name.split('.')[0] + '.png'))


if __name__ == '__main__':
    ####------ Parse arguments from console  ------####
    args = parser.get_arguments()
    main(args)