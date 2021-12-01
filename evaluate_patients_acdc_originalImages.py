# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

# adapted by Carina for RALIS

import os
import glob
import numpy as np
import logging
import torch
from torch.autograd import Variable
from models.model_utils import create_models

import argparse
#import metrics_acdc
import time
#from importlib.machinery import SourceFileLoader
#import tensorflow as tf
from skimage import transform
import nibabel as nib

import utils
import image_utils
import utils.parser as parser



def score_data(input_folder, output_folder, model_path, exp_config, do_postprocessing=False, gt_exists=True, evaluate_all=False, use_iter=None):
    image_size = [256,256]
    nx, ny = image_size[:2]
    batch_size = 1
    num_channels = 4
    target_resolution=(1.36719, 1.36719)

    test_patient_ids = ['011', '012', '013', '014', '015', '016', '017',
                '018', '031', '032', '033', '034', '035', '036', '037', '038', '051', '052',
                '053', '054', '055', '056', '057', '058', '071', '072', '073', '074', '075',
                '076', '077', '078', '091', '092', '093', '094', '095', '096', '097', '098'] # evaluate only those from test split

    kwargs_models = {"dataset": exp_config.dataset,
                    "al_algorithm": exp_config.al_algorithm,
                    "region_size": exp_config.region_size}
    print("kwargs_models: ", kwargs_models)
    net, _, _ = create_models(**kwargs_models)

    #load best model from checkpoint folder
    #print("exp_name: ", exp_config.exp_name)
    #print("ckpt_path: ", exp_config.ckpt_path)
    net_checkpoint_path = os.path.join(exp_config.ckpt_path, exp_config.exp_name, 'best_jaccard_val.pth') #ckpt_path and exp_name from parser
    print("net_checkpoint_path: ", net_checkpoint_path)

    if os.path.isfile(net_checkpoint_path):
        print("net_checkpoint_path: ", net_checkpoint_path)
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

    total_time = 0
    total_volumes = 0

    for folder in os.listdir(input_folder):
        if any([pat_id in folder for pat_id in test_patient_ids]):
            for pat_id in test_patient_ids:
                if pat_id in folder:
                    folder_path = os.path.join(input_folder, folder)
            # '/mnt/qb/baumgartner/cschmidt77_data/acdc_challenge/train/patient001'
            if os.path.isdir(folder_path):
                infos = {}
                for line in open(os.path.join(folder_path, 'Info.cfg')):
                    label, value = line.split(':')
                    infos[label] = value.rstrip('\n').lstrip(' ')

                patient_id = folder.lstrip('patient')
                ED_frame = int(infos['ED'])
                ES_frame = int(infos['ES'])

                
                for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):   
                    logging.info(' ----- Doing image: -------------------------')
                    logging.info('Doing: %s' % file)
                    logging.info(' --------------------------------------------')

                    file_base = file.split('.nii.gz')[0]

                    frame = int(file_base.split('frame')[-1])
                    img_dat = load_nii(file)
                    img = img_dat[0].copy()
                    img = image_utils.normalise_image(img)

                    # load original mask images
                    if gt_exists:
                        file_mask = file_base + '_gt.nii.gz'
                        mask_dat = load_nii(file_mask)
                        mask = mask_dat[0]

                    start_time = time.time()

                    if exp_config.modality == '2D':

                        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
                        scale_vector = (pixel_size[0] / target_resolution[0],
                                        pixel_size[1] / target_resolution[1])

                        predictions = []

                        for zz in range(img.shape[2]):
                            slice_img = np.squeeze(img[:,:,zz])
                            slice_rescaled = transform.rescale(slice_img,
                                                            scale_vector,
                                                            order=1,
                                                            preserve_range=True,
                                                            multichannel=False,
                                                            anti_aliasing=True,
                                                            mode='constant')

                            x, y = slice_rescaled.shape #[235, 278]

                            x_s = (x - nx) // 2
                            y_s = (y - ny) // 2
                            x_c = (nx - x) // 2
                            y_c = (ny - y) // 2
                            
                            # Crop section of image for prediction
                            if x > nx and y > ny:
                                slice_cropped = slice_rescaled[x_s:x_s+nx, y_s:y_s+ny]
                            else:
                                slice_cropped = np.zeros((nx,ny))
                                if x <= nx and y > ny:
                                    slice_cropped[x_c:x_c+x, :] = slice_rescaled[:,y_s:y_s + ny]
                                elif x > nx and y <= ny:
                                    slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]
                                else:
                                    slice_cropped[x_c:x_c+x, y_c:y_c + y] = slice_rescaled[:, :]

                            # GET PREDICTION
                            slice_cropped = np.squeeze(slice_cropped)
                            #print("slice_cropped: ", slice_cropped.shape)
                            input_img = torch.from_numpy(slice_cropped) 
                            input_img = torch.stack((input_img, input_img, input_img), dim=0) #torch.Size([3, 256, 256])
                            if input_img.dim() == 3:
                                img_sz = input_img.size()
                                input_img = input_img.view(1, img_sz[0], img_sz[1], img_sz[2])
                                input_img = Variable(input_img).cuda()
                            #get mask prediction
                            outputs, _ = net(input_img.float()) #floats mask_pred.shape torch.Size([1, 4, 256, 256])
                            #print("outputs.shape: ", outputs.shape) #outputs.shape:  torch.Size([1, 4, 256, 256])
                            predictions_py = outputs
                            #print("predictions_py.shape: ", predictions_py.shape) # torch.Size([1, 256, 256])
                            predictions_py = torch.squeeze(predictions_py) #removes dimension of size 1
                            pred_cpu = predictions_py.cpu()
                            prediction_cropped = np.squeeze(pred_cpu.detach())#.numpy())
                            prediction_cropped = np.transpose(prediction_cropped, (1,2,0)) #[256,256,4]
                            #print("prediction_cropped.shape: ", prediction_cropped.shape)

                            # ASSEMBLE BACK THE SLICES
                            #print("Assemble back the slices")
                            slice_predictions = np.zeros((x,y,num_channels)) #@carina changed num_channels
                            # insert cropped region into original image again
                            if x > nx and y > ny:
                                    slice_predictions[x_s:x_s+nx, y_s:y_s+ny,:] = prediction_cropped
                            else:
                                if x <= nx and y > ny:
                                    slice_predictions[:, y_s:y_s+ny,:] = prediction_cropped[x_c:x_c+ x, :,:]
                                elif x > nx and y <= ny:
                                    slice_predictions[x_s:x_s + nx, :,:] = prediction_cropped[:, y_c:y_c + y,:]
                                else:
                                    slice_predictions[:, :,:] = prediction_cropped[x_c:x_c+ x, y_c:y_c + y,:]

                            # RESCALING ON THE LOGITS
                            #print("check if all elements in array are zero: ", np.all(slice_predictions==0))
                            if gt_exists:
                                #prediction = slice_predictions
                                prediction = transform.resize(slice_predictions, #prediction=[216,256,4], slice_pred=[235, 278,4]
                                                                (mask.shape[0], mask.shape[1], num_channels),#mask.shape [216, 256]
                                                                order=1, 
                                                                preserve_range=True,
                                                                anti_aliasing=True,
                                                                mode='constant')
                            else:  # This can occasionally lead to wrong volume size, therefore if gt_exists
                                    # we use the gt mask size for resizing.
                                prediction = transform.rescale(slice_predictions,
                                                                (1.0/scale_vector[0], 1.0/scale_vector[1], 1),
                                                                order=1,
                                                                preserve_range=True,
                                                                multichannel=False,
                                                                anti_aliasing=True,
                                                                mode='constant')

                            prediction = np.uint8(np.argmax(prediction, axis=-1))
                            print("prediction.shape after resize: ", prediction.shape)
                            #print("check if prediction is all 0: ", np.all(prediction==0))
                            predictions.append(prediction)

                        # import ipdb
                        # ipdb.set_trace()
                        prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))
                    # This is the same for 2D and 3D again
                    if do_postprocessing:
                        prediction_arr = image_utils.keep_largest_connected_components(prediction_arr)

                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    total_volumes += 1

                    logging.info('Evaluation of volume took %f secs.' % elapsed_time)

                    if frame == ED_frame:
                        frame_suffix = '_ED'
                    elif frame == ES_frame:
                        frame_suffix = '_ES'
                    else:
                        raise ValueError('Frame doesnt correspond to ED or ES. frame = %d, ED = %d, ES = %d' %
                                            (frame, ED_frame, ES_frame))

                    # create folder for prediction, ground_truth, image and difference
                    folders = ['prediction', 'ground_truth', 'image', 'difference']
                    for f in folders:
                        if not os.path.exists(os.path.join(output_folder, f)):
                            os.makedirs(os.path.join(output_folder, f))

                    # Save prediced mask
                    out_file_name = os.path.join(output_folder, 'prediction',
                                                    'patient' + patient_id + frame_suffix + '.nii.gz')
                    # if not os.path.exists(out_file_name):
                    #     #print("out_file_name does not exist!")
                    #     os.makedirs(out_file_name) # creats directory instead of file that i don't want

                    if gt_exists:
                        out_affine = mask_dat[1]
                        out_header = mask_dat[2]
                    else:
                        out_affine = img_dat[1]
                        out_header = img_dat[2]

                    logging.info('saving to: %s' % out_file_name)
                    #print("prediction_arr.shape: ", prediction_arr.shape)
                    save_nii(out_file_name, prediction_arr, out_affine, out_header)

                    # Save image data to the same folder for convenience
                    image_file_name = os.path.join(output_folder, 'image',
                                            'patient' + patient_id + frame_suffix + '.nii.gz')

        
                    #print("image_file_name does not exist!")
                    logging.info('saving to: %s' % image_file_name)
                    save_nii(image_file_name, img_dat[0], out_affine, out_header)

                    if gt_exists:

                        # Save GT image
                        gt_file_name = os.path.join(output_folder, 'ground_truth', 'patient' + patient_id + frame_suffix + '.nii.gz')
                        #os.makedirs(gt_file_name)
                        logging.info('saving to: %s' % gt_file_name)
                        save_nii(gt_file_name, mask, out_affine, out_header)

                        # Save difference mask between predictions and ground truth
                        difference_mask = np.where(np.abs(prediction_arr-mask) > 0, [1], [0])
                        difference_mask = np.asarray(difference_mask, dtype=np.uint8)
                        diff_file_name = os.path.join(output_folder,
                                                        'difference',
                                                        'patient' + patient_id + frame_suffix + '.nii.gz')
                        logging.info('saving to: %s' % diff_file_name)
                        save_nii(diff_file_name, difference_mask, out_affine, out_header)
                
        else:
            print("not a test patient")
            continue

        logging.info('Average time per volume: %f' % (total_time/total_volumes))

    #return init_iteration

def load_nii(img_path):
    '''
    Shortcut to load a nifti file
    '''
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    '''
    Shortcut to save a nifty file
    '''

    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


if __name__ == '__main__':
    args = parser.get_arguments()
    root = '/mnt/qb/baumgartner/cschmidt77_data/acdc_challenge/train'
    output_folder =  '/mnt/qb/baumgartner/cschmidt77_data/FINAL_ACDC3232/' + args.exp_name
    # where the predictions of loaded model are located
    model_path = ''

    print("output_folder: ", output_folder)

    score_data(input_folder=root, output_folder=output_folder, model_path=model_path, exp_config=args, do_postprocessing=True,
    gt_exists=True, evaluate_all=True, use_iter=None)

  # singularity exec --nv --bind /mnt/qb/baumgartner ralis.sif python3 -u devel/ralis/evaluate_patients_acdc_originalImages.py --exp-name '2021-10-11-acdc_test_ep49_RIRD_ImageNetBackbone_lr_0.01_budget_128_seed_77' --checkpointer  --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/exp2_acdc_train_ImageNetBackbone'  --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'acdc' --al-algo 'ralis'

  #locally:
  #singularity exec --nv --bind /mnt/qb/baumgartner tue-slurm-helloworld/ralis.sif python3 -u ralis/evaluate_patients_acdc_originalImages.py --exp-name '2021-10-11-acdc_test_ep49_RIRD_ImageNetBackbone_lr_0.01_budget_128_seed_77' --checkpointer  --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/exp2_acdc_train_ImageNetBackbone' --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'acdc' --al-algo 'ralis'

  # singularity exec --nv --bind /mnt/qb/baumgartner ralis.sif python3 -u devel/ralis/evaluate_patients_acdc_originalImages.py --exp-name '2021-10-11-acdc_test_ep49_RIRD_ImageNetBackbone_lr_0.01_budget_128_seed_77' --checkpointer  --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/exp2_acdc_train_ImageNetBackbone'  --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'acdc' --al-algorithm 'ralis'

  # locally:
# singularity exec --nv --bind /mnt/qb/baumgartner tue-slurm-helloworld/ralis.sif python3 -u ralis/evaluate_patients_acdc_originalImages.py --exp-name '2021-11-05-test_acdc_ImageNetBackbone_budget_3568_lr_0.05_3patients_seed_123' --checkpointer  --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/pat2' --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'acdc' --al-algorithm 'ralis'


#singularity exec --nv --bind /mnt/qb/baumgartner tue-slurm-helloworld/ralis.sif python3 -u ralis/evaluate_patients_acdc_originalImages.py --exp-name '2021-11-05-test_acdc_ImageNetBackbone_budget_3568_lr_0.05_3patients_seed_123' --checkpointer  --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/exp4_acdc_train_DT_small' --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'acdc' --al-algorithm 'ralis'