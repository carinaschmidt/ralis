import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

import utils.joint_transforms as joint_transforms
import utils.joint_transforms_acdc as joint_transforms_acdc
import utils.joint_transforms_medical as joint_transforms_medical
import utils.transforms as extended_transforms
from data import cityscapes, acdc, acdc_al, msdHeart, brats18_2D, brats18_2D_al, brats18, brats18_al
import utils.parser as parser
import torchio as tio

def get_data(data_path, code_path, tr_bs, vl_bs, n_workers=0, scale_size=0, input_size=(256, 512),
             supervised=False, num_each_iter=1, only_last_labeled=False, dataset='cityscapes', test=False,
             al_algorithm='ralis', full_res=False,
             region_size=128):
    print('Loading data...')
    args = parser.get_arguments()
    candidate_set = None
    input_transform, target_transform, train_joint_transform, val_joint_transform, al_train_joint_transform, train_joint_transform_region_3D = \
    get_transforms(scale_size, input_size, region_size, supervised, test, al_algorithm, full_res, dataset)

    if supervised:
        if 'acdc' in dataset:
            args = parser.get_arguments()
            if "allPatients" in args.exp_name:
                subset = False
            else:
                subset = True
                print("subset=True")
            train_set = acdc.ACDC('fine', 'train',
                                      data_path=data_path,
                                      code_path=code_path,
                                      joint_transform=train_joint_transform,
                                      transform=input_transform,
                                      target_transform=target_transform, subset=subset)
            val_set = acdc.ACDC('fine', 'val',
                                    data_path=data_path,
                                    code_path=code_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
        # MSD Heart for pre-Training 
        elif dataset == 'msdHeart':
            train_set = msdHeart.MSD_Heart('fine', 'train',
                                      data_path=data_path,
                                      code_path=code_path,
                                      joint_transform=val_joint_transform,
                                      transform=input_transform,
                                      target_transform=target_transform, subset=False)
            print("For training using: ", train_joint_transform)
            val_set = msdHeart.MSD_Heart('fine', 'val',
                                    data_path=data_path,
                                    code_path=code_path,
                                    joint_transform=train_joint_transform, #usually val_joint_transform! just to test if train/val loss changes
                                    transform=input_transform,
                                    target_transform=target_transform, subset=False)
            print("For validation using: ", train_joint_transform)

        elif dataset == 'brats18':
            if args.modality == '2D':
                print("Train set for Brats18 2D: ")
                if "allPatients" in args.exp_name:
                    subset = False
                else:
                    subset = True

                train_set = brats18_2D.BraTS18_2D('fine', 'train',
                                        data_path=data_path,
                                        code_path=code_path,
                                        joint_transform=train_joint_transform,
                                        transform=input_transform,
                                        target_transform=target_transform, subset=subset)
                print("length of train_set: ", len(train_set))
                
                val_set = brats18_2D.BraTS18_2D('fine', 'val',
                                    data_path=data_path,
                                    code_path=code_path,
                                    joint_transform=val_joint_transform,
                                    transform=input_transform,
                                    target_transform=target_transform)
                print("length of val_set: ", len(val_set))
            else: #3D
                print("Train set for Brats18 3D: ")
                train_set = brats18.Brats18(mode='train', data_path=data_path,
                                            code_path=code_path, subset = False)
                print("Candidate_set: ")
                print("Val_set 3D: ")
                val_set = brats18.Brats18(mode='val',  data_path=data_path,
                                            code_path=code_path, subset = False) 


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
        if dataset == 'acdc':
            # d_V! if "test"
            # mode is 'train', split is 'test' for slurm_test.sh, 
            # train_set used to train the segmentation net
            print("train_joint_transform: ", train_joint_transform)
            print("Train_set: ")
            train_set = acdc_al.ACDC_al('fine', 'train',
                                            data_path=data_path,
                                            code_path=code_path,
                                            joint_transform=train_joint_transform,
                                            #joint_transform_region = train_joint_transform_region,
                                            joint_transform_acdc_al = al_train_joint_transform, #train_joint_transform_acdc_al,
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
            print("len of train set: ", len(train_set))
            print("len of candidate set: ", len(candidate_set))
            print("len of val_set: ", len(val_set))
        elif dataset == 'brats18':
            if args.modality == '2D':
                print("train set for brats18 2D: ")
                train_set = brats18_2D_al.BraTS18_2D_al('fine', 'train',
                                            data_path=data_path,
                                            code_path=code_path,
                                            joint_transform=train_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform, num_each_iter=num_each_iter,
                                            only_last_labeled=only_last_labeled,
                                            split='train' if al_algorithm == 'ralis' and not test else 'test',
                                            region_size=region_size)
              
                print("length of train set: ", len(train_set))
                print("candidate set for brats18 2D: ")
                candidate_set = brats18_2D_al.BraTS18_2D_al('fine', 'train',
                                                data_path=data_path,
                                                code_path=code_path,
                                                joint_transform=None,
                                                candidates_option=True,
                                                transform=input_transform,
                                                target_transform=target_transform,
                                                split='train' if al_algorithm == 'ralis' and not test else 'test',
                                                region_size=region_size)
                print("length of candidate set: ", len(candidate_set))
                val_set = brats18_2D.BraTS18_2D('fine', 'val',
                                            data_path=data_path,
                                            code_path=code_path,
                                            joint_transform=val_joint_transform,
                                            transform=input_transform,
                                            target_transform=target_transform)
                print("length of val_set for brats18: ", len(val_set))
        
            else: #3D
                print("Train set for Brats: ")
                train_set = brats18_al.BraTS18_al(mode='train', joint_transform_region=train_joint_transform_region_3D)
                print("Candidate_set: ")
                # mode is 'train', split is 'test' (default) -> d_v 
                # candidate_set: image indexes to be potentially labeled @carina 
                candidate_set = brats18_al.BraTS18_al(mode='test')
                print("Val_set: ")
                val_set = brats18_al.BraTS18_al(mode='val')

    print("before train loader: ", len(train_set))
    print("before val loader: ", len(val_set))
    train_loader = DataLoader(train_set,
                              batch_size=tr_bs,
                              num_workers=n_workers, shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(val_set,
                            batch_size=vl_bs,
                            num_workers=n_workers, shuffle=False)
    return train_loader, train_set, val_loader, candidate_set


def get_transforms(scale_size, input_size, region_size, supervised, test, al_algorithm, full_res, dataset):
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_joint_transform_region_3D = None
    train_joint_transform = None
    val_joint_transform = None
    al_train_joint_transform =  None

    args = parser.get_arguments()
    
    if args.modality == '2D' and (dataset == 'acdc' or dataset == 'msdHeart' or dataset == 'brats18'):
        if scale_size == 0:
            print('(Data loading) Not scaling the data by size')
            print('(Data loading) Random crops of ' + str(input_size) + ' in training')
            print('(Data loading) No crops in validation')
            if supervised:
                print("in supervised")
                # new augmentations                
                if args.newAugmentations: # use GI/GD
                    print("using new augmentations RI/RD")
                    if 'acdc' in dataset: 
                        train_joint_transform = joint_transforms_acdc.Compose([
                            #joint_transforms_medical.DoubleCropOrPad(input_size),
                            joint_transforms_acdc.RandomCrop(input_size),
                            joint_transforms_acdc.RandomHorizontallyFlip02(),
                            #joint_transforms_acdc.RandomHorizontallyFlip(),
                            joint_transforms_medical.DoubleRandomRotate(p=0.2), #scaling
                            joint_transforms_medical.DoubleRandomScale(p=0.2),
                            joint_transforms_medical.DoubleRand2DElastic(p=0.2, input_size=input_size), #0.2
                            joint_transforms_medical.ContrastBrightnessAdjustment(p=0.2) #0.2
                            # 0.5 random flip, 0.25 elastic und contrast/brightness
                            ])
                    elif 'brats18' in dataset:
                        train_joint_transform = joint_transforms_medical.Compose([
                            joint_transforms_medical.DoubleCropOrPad(input_size), #from MonAI: 
                            joint_transforms_medical.DoubleRandomRotate(p=0.2), #scaling
                            joint_transforms_medical.DoubleRandomScale(p=0.2),
                            joint_transforms_medical.DoubleHorizontalFlip(p=0.2),
                            joint_transforms_medical.DoubleRand2DElastic(p=0.2, input_size=input_size),
                            joint_transforms_medical.ContrastBrightnessAdjustment(p=0.2)
                            ])
                    else:
                        print("specify correct dataset")

                else:
                    print("using old standard augmentations")
                    #old transforms
                    if 'acdc' in dataset: 
                        train_joint_transform = joint_transforms_acdc.Compose([
                            joint_transforms_acdc.RandomCrop(input_size),
                            joint_transforms_acdc.RandomHorizontallyFlip()
                        ])
                    elif 'brats18' in dataset:
                        train_joint_transform = joint_transforms_medical.Compose([
                            joint_transforms_medical.DoubleCropOrPad(input_size),
                            joint_transforms_medical.DoubleHorizontalFlip(p=0.5)
                        ])

            else: #not supervised (with crop region instead of crop or pad)       
                if args.newAugmentations:
                    if 'acdc' in dataset:
                        print("using new augmentations RI/RD")
                        train_joint_transform = joint_transforms_acdc.ComposeRegion([
                                joint_transforms_acdc.RandomCropRegion(input_size, region_size=region_size), 
                                joint_transforms_acdc.RandomHorizontallyFlip02(),
                                joint_transforms_medical.DoubleRandomRotate(p=0.2),
                                joint_transforms_medical.DoubleRandomScale(p=0.2),
                                joint_transforms_medical.DoubleRand2DElastic(p=0.2, input_size=input_size),
                                joint_transforms_medical.ContrastBrightnessAdjustment(p=0.2)
                                # 0.5 random flip, 0.25 elastic und contrast/brightness
                            ])
                    elif 'brats18' in dataset:
                       train_joint_transform = joint_transforms_medical.Compose([
                                joint_transforms_medical.DoubleCropRandomRegion(input_size, region_size=region_size),
                                joint_transforms_medical.DoubleRandomRotate(p=0.2),
                                joint_transforms_medical.DoubleRandomScale(p=0.2),
                                joint_transforms_medical.DoubleHorizontalFlip(p=0.2),
                                joint_transforms_medical.DoubleRand2DElastic(p=0.2, input_size=input_size),
                                joint_transforms_medical.ContrastBrightnessAdjustment(p=0.2)
                        ])

                else:
                    print("using standard augmentations")
                    # ralis augmentations
                    if 'acdc' in dataset:
                        train_joint_transform = joint_transforms_acdc.ComposeRegion([
                            joint_transforms_acdc.RandomCropRegion(input_size, region_size=region_size), 
                            joint_transforms_acdc.RandomHorizontallyFlip()
                        ])
                    elif 'brats18' in dataset:
                        train_joint_transform = joint_transforms_medical.Compose([
                            joint_transforms_medical.DoubleCropRandomRegion(input_size, region_size=region_size),
                            joint_transforms_medical.DoubleHorizontalFlip(p=0.5),
                        ])
               
            if (not test and al_algorithm == 'ralis') and not full_res:
                if 'acdc' in dataset: 
                    scale_size = [256, 256]
                    val_joint_transform = joint_transforms_acdc.Scale(256)
                elif 'brats' in dataset:
                    scale_size = [160, 192]
                else: 
                    print("specify scale size")
            else:
                val_joint_transform = None
        
        else: # scale sized not 0
            print('(Data loading) Scaling training data: ' + str(
                scale_size) + ' width dimension')
            print('(Data loading) Random crops of ' + str(
                input_size) + ' in training')
            print('(Data loading) No crops nor scale_size in validation')
            if supervised:
                train_joint_transform = joint_transforms_acdc.Compose([
                     joint_transforms_medical.DoubleScale(scale_size),
                     joint_transforms_medical.DoubleRandomRotate(p=0.2),
                     joint_transforms_medical.DoubleRandomScale(p=0.2),
                     joint_transforms_medical.DoubleHorizontalFlip(p=0.2),
                     joint_transforms_medical.DoubleRand2DElastic(p=0.2, input_size=input_size),
                     joint_transforms_medical.ContrastBrightnessAdjustment(p=0.2)
                    ])
           
            else:
                train_joint_transform = tio.Compose([
                    tio.CropOrPad((region_size[0], region_size[1], 1)),
                    tio.RandomFlip(axes='A'), #horizontal flip
                    tio.OneOf({
                    tio.RandomAffine(scales=(0.8,1.2), degrees=(-15, 15), default_pad_value='minimum', image_interpolation='nearest'): 0.7,
                    tio.RandomElasticDeformation(num_control_points=4, max_displacement=0.5, locked_borders=0): 0.2,
                    tio.RandomGamma(log_gamma=(-0.3, 0.3)): 0.1
                    })
                ])

            al_train_joint_transform = tio.Compose([
                    tio.CropOrPad((region_size[0], region_size[1], 1)),
                    tio.RandomFlip(axes='A'), #horizontal flip
                    tio.OneOf({
                    tio.RandomAffine(scales=(0.8,1.2), degrees=(-15, 15), default_pad_value='minimum', image_interpolation='nearest'): 0.7,
                    tio.RandomElasticDeformation(num_control_points=4, max_displacement=0.5, locked_borders=0): 0.2,
                    tio.RandomGamma(log_gamma=(-0.3, 0.3)): 0.1
                    })
                ])


            val_joint_transform = None

    else: # either not 2D data or not ACDC or BraTS data
        train_joint_transform_region = None
        train_joint_transform_acdc_al = None
        train_joint_transform_region_3D = None
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
                val_joint_transform = joint_transforms.Scale(256)
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
    #train_joint_transform = None, # NO augmentations !! 
    target_transform = None, 
    input_transform = None
    #val_joint_transform = None
    return input_transform, target_transform, train_joint_transform, val_joint_transform, al_train_joint_transform, train_joint_transform_region_3D
