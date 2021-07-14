import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

import utils.joint_transforms as joint_transforms
import utils.joint_transforms_acdc as joint_transforms_acdc
import utils.transforms as extended_transforms
import utils.transforms_acdc as extended_transforms_acdc
from data import cityscapes, gtav, cityscapes_al, cityscapes_al_splits, camvid, camvid_al, acdc, acdc_al, acdc_al_splits
import numpy as np

def get_data(data_path, code_path, tr_bs, vl_bs, n_workers=0, scale_size=0, input_size=(256, 512),
             supervised=False, num_each_iter=1, only_last_labeled=False, dataset='cityscapes', test=False,
             al_algorithm='ralis', full_res=False,
             region_size=128):
    print('Loading data...')
    candidate_set = None
    #import ipdb
    #ipdb.set_trace()
    input_transform, target_transform, train_joint_transform, val_joint_transform, al_train_joint_transform = \
        get_transforms(scale_size, input_size, region_size, supervised, test, al_algorithm, full_res, dataset)

    # To train pre-trained segmentation network and upper bounds.
    if supervised:
        if 'gta' in dataset:
            train_set = gtav.GTAV('fine', 'train',
                                  data_path=data_path,
                                  joint_transform=train_joint_transform,
                                  transform=input_transform,
                                  target_transform=target_transform,
                                  camvid=True if dataset == 'gta_for_camvid' else False)
            val_set = gtav.GTAV('fine', 'val',
                                data_path=data_path,
                                joint_transform=val_joint_transform,
                                transform=input_transform,
                                target_transform=target_transform,
                                camvid=True if dataset == 'gta_for_camvid' else False)
        elif dataset == 'camvid':
            train_set = camvid.Camvid('fine', 'train',
                                      data_path=data_path,
                                      code_path=code_path,
                                      joint_transform=train_joint_transform,
                                      transform=input_transform,
                                      target_transform=target_transform)
            val_set = camvid.Camvid('fine', 'val',
                                    data_path=data_path,
                                    code_path=code_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
        elif dataset == 'camvid_subset':
            train_set = camvid.Camvid('fine', 'train',
                                      data_path=data_path,
                                      code_path=code_path,
                                      joint_transform=train_joint_transform,
                                      transform=input_transform,
                                      target_transform=target_transform, subset=True)
            val_set = camvid.Camvid('fine', 'val',
                                    data_path=data_path,
                                    code_path=code_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
        # @carina added acdc part for pre-training of segmentation network!
        elif dataset == 'acdc':
            train_set = acdc.ACDC('fine', 'train',
                                      data_path=data_path,
                                      code_path=code_path,
                                      joint_transform=train_joint_transform,
                                      transform=input_transform,
                                      target_transform=target_transform, subset=True)
            # train_set = acdc.ACDC('fine', 'train',
            #                           data_path=data_path,
            #                           code_path=code_path,
            #                           joint_transform=train_joint_transform,
            #                           transform=input_transform,
            #                           target_transform=target_transform, subset=False) #using all train data
            val_set = acdc.ACDC('fine', 'val',
                                    data_path=data_path,
                                    code_path=code_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
        elif dataset == 'cs_upper_bound':
            train_set = cityscapes_al_splits.CityScapes_al_splits('fine', 'train',
                                                                  data_path=data_path,
                                                                  code_path=code_path,
                                                                  joint_transform=train_joint_transform,
                                                                  transform=input_transform,
                                                                  target_transform=target_transform, supervised=True)
            val_set = cityscapes.CityScapes('fine', 'val',
                                            data_path=data_path,
                                            code_path=code_path,
                                            joint_transform=val_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform)

        elif dataset == 'cityscapes_subset':
            train_set = cityscapes_al_splits.CityScapes_al_splits('fine', 'train',
                                                                  data_path=data_path,
                                                                  code_path=code_path,
                                                                  joint_transform=train_joint_transform,
                                                                  transform=input_transform,
                                                                  target_transform=target_transform, subset=True)
            val_set = cityscapes.CityScapes('fine', 'val',
                                            data_path=data_path,
                                            code_path=code_path,
                                            joint_transform=val_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform)

        else:
            train_set = cityscapes.CityScapes('fine', 'train',
                                              data_path=data_path,
                                              code_path=code_path,
                                              joint_transform=train_joint_transform,
                                              transform=input_transform,
                                              target_transform=target_transform)
            val_set = cityscapes.CityScapes('fine', 'val',
                                            data_path=data_path,
                                            code_path=code_path,
                                            joint_transform=val_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform)
    # To train AL methods
    else:
        if dataset == 'cityscapes':
            if al_algorithm == 'ralis' and not test:
                split = 'train'
            else:
                split = 'test'
            train_set = cityscapes_al.CityScapes_al('fine', 'train',
                                                    data_path=data_path,
                                                    code_path=code_path,
                                                    joint_transform=train_joint_transform,
                                                    joint_transform_al=al_train_joint_transform,
                                                    transform=input_transform,
                                                    target_transform=target_transform, num_each_iter=num_each_iter,
                                                    only_last_labeled=only_last_labeled,
                                                    split=split, region_size=region_size)
            
            candidate_set = cityscapes_al.CityScapes_al('fine', 'train',
                                                        data_path=data_path,
                                                        code_path=code_path,
                                                        joint_transform=None,
                                                        candidates_option=True,
                                                        transform=input_transform,
                                                        target_transform=target_transform, split=split,
                                                        region_size=region_size)
            # no 'split' in _al_splits train  -> d_r!
            val_set = cityscapes_al_splits.CityScapes_al_splits('fine', 'train',
                                                                data_path=data_path,
                                                                code_path=code_path,
                                                                joint_transform=val_joint_transform,
                                                                transform=input_transform,
                                                                target_transform=target_transform)

        elif dataset == 'camvid':
            train_set = camvid_al.Camvid_al('fine', 'train',
                                            data_path=data_path,
                                            code_path=code_path,
                                            joint_transform=train_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            only_last_labeled=only_last_labeled,
                                            split='train' if al_algorithm == 'ralis' and not test else 'test',
                                            region_size=region_size)
            candidate_set = camvid_al.Camvid_al('fine', 'train',
                                                data_path=data_path,
                                                code_path=code_path,
                                                joint_transform=None,
                                                candidates_option=True,
                                                transform=input_transform,
                                                target_transform=target_transform,
                                                split='train' if al_algorithm == 'ralis' and not test else 'test',
                                                region_size=region_size)

            val_set = camvid.Camvid('fine', 'val',
                                    data_path=data_path,
                                    code_path=code_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
        # @carina added ACDC
        elif dataset == 'acdc':
            #import ipdb
            #ipdb.set_trace()
            # d_V! if "test"
            # mode is 'train', split is 'test' for slurm_test.sh, 
            # train_set used to train the segmentation net
            print("Train_set: ")
            train_set = acdc_al.ACDC_al('fine', 'train',
                                            data_path=data_path,
                                            code_path=code_path,
                                            joint_transform=train_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            only_last_labeled=only_last_labeled,
                                            split='train' if al_algorithm == 'ralis' and not test else 'test', #if --train and --test, still test=TRUE
                                            region_size=region_size)
            print("Candidate_set: ")
            # mode is 'train', split is 'test' (default) -> d_v 
            # candidate_set: image indexes to be potentially labeled @carina 
            candidate_set = acdc_al.ACDC_al('fine', 'train',
                                            data_path=data_path,
                                            code_path=code_path,
                                            joint_transform=None,
                                            candidates_option=True,
                                            transform=input_transform,
                                            target_transform=target_transform,
                                            split='train' if al_algorithm == 'ralis' and not test else 'test',
                                            region_size=region_size)
            print("Val_set: ")
            val_set = acdc.ACDC('fine', 'val',
                                            data_path=data_path,
                                            code_path=code_path,
                                            joint_transform=val_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform)

            #@ added to test if it works with ACDC_al_splits class
            # val_set = acdc_al_splits.ACDC_al_splits('fine', 'val',
            #                                                     data_path=data_path,
            #                                                     code_path=code_path,
            #                                                     joint_transform=val_joint_transform,
            #                                                     transform=input_transform,
            #                                                     target_transform=target_transform)

    #import ipdb
    #ipdb.set_trace()
    # import ipdb
    # ipdb.set_trace()
    train_loader = DataLoader(train_set,
                              batch_size=tr_bs,
                              num_workers=n_workers, shuffle=True,
                              drop_last=False)

    print("Length of train loader: ", len(train_loader))

    val_loader = DataLoader(val_set,
                            batch_size=vl_bs,
                            num_workers=n_workers, shuffle=False)
    
    print("Length of val loader: ", len(val_loader))

    return train_loader, train_set, val_loader, candidate_set


def get_transforms(scale_size, input_size, region_size, supervised, test, al_algorithm, full_res, dataset):
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if dataset == 'acdc':
        #import ipdb
        #ipdb.set_trace()
        print("dataset: acdc")
        print("input_size in DataUtils: ", input_size)
        if scale_size == 0:
            print('(Data loading) Not scaling the data')
            print('(Data loading) Random crops of ' + str(input_size) + ' in training')
            print('(Data loading) No crops in validation')
            if supervised:
                train_joint_transform = joint_transforms_acdc.Compose([
                    joint_transforms_acdc.RandomCrop(input_size),
                    joint_transforms_acdc.RandomHorizontallyFlip()
                ])
            else:
                train_joint_transform = joint_transforms_acdc.ComposeRegion([
                    joint_transforms_acdc.RandomCropRegion(input_size, region_size=region_size),
                    joint_transforms_acdc.RandomHorizontallyFlip()
                ])
            if (not test and al_algorithm == 'ralis') and not full_res:
                val_joint_transform = joint_transforms_acdc.Scale(256) #TODO
            else:
                val_joint_transform = None
            al_train_joint_transform = joint_transforms_acdc.ComposeRegion([
                joint_transforms_acdc.CropRegion(region_size, region_size=region_size),
                joint_transforms_acdc.RandomHorizontallyFlip()
            ])
        else:
            print('(Data loading) Scaling training data: ' + str(
                scale_size) + ' width dimension')
            print('(Data loading) Random crops of ' + str(
                input_size) + ' in training')
            print('(Data loading) No crops nor scale_size in validation')
            if supervised:
                train_joint_transform = joint_transforms.Compose([
                    joint_transforms_acdc.Scale(scale_size),
                    joint_transforms_acdc.RandomCrop(input_size),
                    joint_transforms_acdc.RandomHorizontallyFlip()
                ])
            else:
                train_joint_transform = joint_transforms.ComposeRegion([
                    joint_transforms_acdc.Scale(scale_size),
                    joint_transforms_acdc.RandomCropRegion(input_size, region_size=region_size),
                    joint_transforms_acdc.RandomHorizontallyFlip()
                ])
            al_train_joint_transform = joint_transforms_acdc.ComposeRegion([
                joint_transforms_acdc.Scale(scale_size),
                joint_transforms_acdc.CropRegion(region_size, region_size=region_size),
                joint_transforms_acdc.RandomHorizontallyFlip()
            ])
            if dataset == 'gta_for_camvid':
                val_joint_transform = joint_transforms_acdc.ComposeRegion([
                    joint_transforms_acdc.Scale(scale_size)])
            else:
                val_joint_transform = None
        input_transform = standard_transforms.Compose([
            extended_transforms_acdc.ImageToTensor() #@carina added
            #standard_transforms.Normalize(*mean_std)
            # TODO normalization
        ])
        target_transform = extended_transforms_acdc.MaskToTensor()  #TODO
        

    else:
        if scale_size == 0:
            print('(Data loading) Not scaling the data')
            print('(Data loading) Random crops of ' + str(input_size) + ' in training')
            print('(Data loading) No crops in validation')
            
            if supervised:
                train_joint_transform = joint_transforms.Compose([
                    joint_transforms.RandomCrop(input_size),
                    joint_transforms.RandomHorizontallyFlip()
                ])
            else:
                train_joint_transform = joint_transforms.ComposeRegion([
                    joint_transforms.RandomCropRegion(input_size, region_size=region_size),
                    joint_transforms.RandomHorizontallyFlip()
                ])
            if (not test and al_algorithm == 'ralis') and not full_res:
                val_joint_transform = joint_transforms.Scale(1024) #TODO change 1024
            else:
                val_joint_transform = None
            al_train_joint_transform = joint_transforms.ComposeRegion([
                joint_transforms.CropRegion(region_size, region_size=region_size),
                joint_transforms.RandomHorizontallyFlip()
            ])
        else:
            print('(Data loading) Scaling training data: ' + str(
                scale_size) + ' width dimension')
            print('(Data loading) Random crops of ' + str(
                input_size) + ' in training')
            print('(Data loading) No crops nor scale_size in validation')
            if supervised:
                train_joint_transform = joint_transforms.Compose([
                    joint_transforms.Scale(scale_size),
                    joint_transforms.RandomCrop(input_size),
                    joint_transforms.RandomHorizontallyFlip()
                ])
            else:
                train_joint_transform = joint_transforms.ComposeRegion([
                    joint_transforms.Scale(scale_size),
                    joint_transforms.RandomCropRegion(input_size, region_size=region_size),
                    joint_transforms.RandomHorizontallyFlip()
                ])
            al_train_joint_transform = joint_transforms.ComposeRegion([
                joint_transforms.Scale(scale_size),
                joint_transforms.CropRegion(region_size, region_size=region_size),
                joint_transforms.RandomHorizontallyFlip()
            ])
            if dataset == 'gta_for_camvid':
                val_joint_transform = joint_transforms.ComposeRegion([
                    joint_transforms.Scale(scale_size)])
            else:
                val_joint_transform = None
        input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        target_transform = extended_transforms.MaskToTensor() 

    return input_transform, target_transform, train_joint_transform, val_joint_transform, al_train_joint_transform