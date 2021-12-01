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
import torch.nn.functional as F

def main(args):
    ####------ Create segmentation, query and target networks ------####
    kwargs_models = {"dataset": args.dataset,
                "al_algorithm": args.al_algorithm,
                "region_size": args.region_size,

            }

    #segmentation net, policy net
    net,_,_ = create_models(**kwargs_models)
    net.eval()

    net_checkpoint_path = os.path.join(args.ckpt_path, args.exp_name, 'best_jaccard_val.pth') #'/mnt/qb/baumgartner/cschmidt77_data/ckpt_seg_new/pretraining_acdc_06-19/last_jaccard_val.pth'
    net_dict = torch.load(net_checkpoint_path)
    net.load_state_dict(net_dict)
    root = os.path.join(args.data_path, args.dataset) 

    print("os.getcwd", os.getcwd())
    d_s = np.load('ralis/data/acdc_pat_img_splits.npy', allow_pickle=True).item()['d_s']

    root = '/mnt/qb/baumgartner/cschmidt77_data/acdc'
    output_folder = '/home/carina/entropy_ds'
    img_path = os.path.join(root, "slices", "train")
    mask_path = os.path.join(root, "gt", "train")
    entropies = []
    pred_slices = []
    for sl in d_s:
        img_path_new = os.path.join(img_path, sl)
        mask_path_new = os.path.join(mask_path, sl)
        img, mask = np.load(img_path_new), np.load(mask_path_new)
        img, mask = torch.from_numpy(img), torch.from_numpy(mask)
        #image = np.transpose(img, (2,0,1))
        #input_img = torch.from_numpy(image) #torch.Size([4, 256, 256])
        #print("input_img: ", input_img)
        input_img = torch.stack((img, img, img), dim=0)
        if input_img.dim() == 3:
            img_sz = input_img.size()
            input_img = input_img.view(1, img_sz[0], img_sz[1], img_sz[2])
            input_img = Variable(input_img).cuda()
        else:
            print("input_img.dim: ", input_img.dim())
        outputs, _ = net(input_img.float())
        #print("before output.data: ",outputs.min())
        print(outputs.min())
        print(outputs.max())
        #predictions_py = torch.squeeze(outputs) #removes dimension of size 1
        pred_cpu = outputs.cpu()
        pred = np.squeeze(pred_cpu.detach())#[1,4,160,192]

        # get pixel-wise class predictions
        pred_py = F.softmax(outputs, dim=1).data
        pred_py = pred_py.max(1)
        predictions_py = pred_py[1].squeeze_(1).cpu().type(torch.FloatTensor)
        # compute entropy:
        ent = compute_entropy_seg(args, input_img, net)
        entropies.append(ent)

        name = 'entropy_%s' %(sl)
        out_file_name = os.path.join(output_folder, 'entropy', name)
        np.save(out_file_name, ent, allow_pickle=True, fix_imports=True)

        name = 'slice_%s' %(sl)
        pred_slices.append(predictions_py)
        out_file_name = os.path.join(output_folder, 'prediction', name)
        np.save(out_file_name, predictions_py, allow_pickle=True, fix_imports=True)

        name = 'slice_%s' %(sl)
        out_file_name = os.path.join(output_folder, 'slice', name)
        np.save(out_file_name,sl, allow_pickle=True, fix_imports=True)




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


if __name__ == '__main__':
    ####------ Parse arguments from console  ------####
    args = parser.get_arguments()
    main(args)

    #singularity exec --nv --bind /mnt/qb/baumgartner tue-slurm-helloworld/ralis.sif python3 -u ralis/visualise_entropy_map.py --exp-name '2021-10-26-train_acdc_ImageNetBackbone_budget_192_lr_0.05_3patients_seed234' --checkpointer  --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/exp4_acdc_train_DT_small/' --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'acdc' --al-algorithm 'ralis'
