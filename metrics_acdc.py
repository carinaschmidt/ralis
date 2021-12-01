"""
Code for evaluation of acdc metrics. Writes full report of experiment performance.

Authors:
Christian F. Baumgartner (c.f.baumgartner@gmail.com)
Lisa. M. Koch (lisa.margret.koch@gmail.com)

Extended by Carina

Extended from code made available by
author: Clément Zotti (clement.zotti@usherbrooke.ca)
date: April 2017
Link: http://acdc.creatis.insa-lyon.fr

"""

import os
from glob import glob
import re
import pandas as pd
from medpy.metric.binary import hd, dc, assd
import numpy as np



import matplotlib.pyplot as plt
import seaborn as sns

import logging

import nibabel as nib
import utils.parser as parser


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

#
# Utils functions used to sort strings into a natural order
#
def conv_int(i):
    return int(i) if i.isdigit() else i



def natural_order(sord):
    """
    Sort a (list,tuple) of strings into natural order.

    Ex:

    ['1','10','2'] -> ['1','2','10']

    ['abc1def','ab10d','b2c','ab1d'] -> ['ab1d','ab10d', 'abc1def', 'b2c']

    """
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


def compute_metrics_on_directories_raw(dir_gt, dir_pred):
    """
    Calculates a number of measures from the predicted and ground truth segmentations:
    - Dice
    - Hausdorff distance
    - Average surface distance
    - Predicted volume
    - Volume error w.r.t. ground truth
    :param dir_gt: Directory of the ground truth segmentation maps.
    :param dir_pred: Directory of the predicted segmentation maps.
    :return: Pandas dataframe with all measures in a row for each prediction and each structure
    """

    filenames_gt = sorted(glob(os.path.join(dir_gt, '*')), key=natural_order)
    filenames_pred = sorted(glob(os.path.join(dir_pred, '*')), key=natural_order)

    cardiac_phase = []
    file_names = []
    structure_names = []

    # 5 measures per structure:
    dices_list = []
    hausdorff_list = []
    assd_list = []
    vol_list = []
    vol_err_list = []

    structures_dict = {1: 'RV', 2: 'MYO', 3: 'LV'}

    for p_gt, p_pred in zip(filenames_gt, filenames_pred):
        # import ipdb 
        # ipdb.set_trace()
        if os.path.basename(p_gt) != os.path.basename(p_pred):
            raise ValueError("The two files don't have the same name"
                             " {}, {}.".format(os.path.basename(p_gt),
                                               os.path.basename(p_pred)))

        # load ground truth and prediction
        gt, _, header = load_nii(p_gt)
        pred, _, _ = load_nii(p_pred)
        zooms = header.get_zooms()
        gt = np.uint8(gt)
        pred = np.uint8(pred)

        # calculate measures for each structure
        for struc in [3,1,2]:
            # import ipdb
            # ipdb.set_trace()

            gt_binary = (gt == struc) * 1
            pred_binary = (pred == struc) * 1

            volpred = pred_binary.sum() * np.prod(zooms) / 1000.
            volgt = gt_binary.sum() * np.prod(zooms) / 1000.

            vol_list.append(volpred)
            vol_err_list.append(volpred - volgt)

            # import ipdb
            # ipdb.set_trace()
            if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
                dices_list.append(1)
                assd_list.append(0)
                hausdorff_list.append(0)
            elif np.sum(pred_binary) > 0 and np.sum(gt_binary) == 0 or np.sum(pred_binary) == 0 and np.sum(gt_binary) > 0:
                logging.warning('Structure missing in either GT (x)or prediction. ASSD and HD will not be accurate.')
                dices_list.append(0)
                assd_list.append(1)
                hausdorff_list.append(1)
            else:
                hausdorff_list.append(hd(gt_binary, pred_binary, voxelspacing=zooms, connectivity=1))
                assd_list.append(assd(pred_binary, gt_binary, voxelspacing=zooms, connectivity=1))
                dices_list.append(dc(gt_binary, pred_binary))

            cardiac_phase.append(os.path.basename(p_gt).split('.nii.gz')[0].split('_')[-1])
            file_names.append(os.path.basename(p_pred))
            structure_names.append(structures_dict[struc])


    df = pd.DataFrame({'3D Dice': dices_list, 'HD': hausdorff_list, 'ASSD': assd_list,
                       'vol': vol_list, 'vol_err': vol_err_list,
                      'phase': cardiac_phase, 'Cardiac structure': structure_names, 'filename': file_names})

    return df


def print_latex_tables(df, eval_dir):
    """
    Report geometric measures in latex tables to be used in the ACDC challenge paper.
    Prints mean (+- std) values for Dice and ASSD for all structures.
    :param df:
    :param eval_dir:
    :return:
    """

    out_file = os.path.join(eval_dir, 'latex_tables.txt')

    with open(out_file, "w") as text_file:

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('ACDC challenge paper: table 1\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')
        # prints mean (+- std) values for Dice and ASSD, all structures, averaged over both phases.

        header_string = ' & '
        line_string = 'METHOD '


        for s_idx, struc_name in enumerate(['LV', 'RV', 'MYO']):
            for measure in ['3D Dice', 'ASSD']:

                header_string += ' & {} ({}) '.format(measure, struc_name)

                dat = df.loc[df['Cardiac structure'] == struc_name]

                if measure == '3D Dice':
                    line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
                else:
                    line_string += ' & {:.2f}\,({:.2f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))

            if s_idx < 2:
                header_string += ' & '
                line_string += ' & '

        header_string += ' \\\\ \n'
        line_string += ' \\\\ \n'

        text_file.write(header_string)
        text_file.write(line_string)


        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('ACDC challenge paper: table 2\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')
        # table 2: mean (+- std) values for Dice, ASSD and HD, all structures, both phases separately


        for idx, struc_name in enumerate(['LV', 'RV', 'MYO']):
            # new line
            header_string = ' & '
            line_string = '({}) '.format(struc_name)

            for p_idx, phase in enumerate(['ED', 'ES']):
                for measure in ['3D Dice', 'ASSD', 'HD']:

                    header_string += ' & {} ({}) '.format(phase, measure)

                    dat = df.loc[(df['phase'] == phase) & (df['Cardiac structure'] == struc_name)]

                    if measure == '3D Dice':

                        line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
                    else:
                        line_string += ' & {:.2f}\,({:.2f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))

                if p_idx == 0:
                    header_string += ' & '
                    line_string += ' & '

            header_string += ' \\\\ \n'
            line_string += ' \\\\ \n'

            if idx == 0:
                text_file.write(header_string)

            text_file.write(line_string)

    return 0


def boxplot_metrics(df, eval_dir):
    """
    Create summary boxplots of all geometric measures.
    :param df:
    :param eval_dir:
    :return:
    """

    boxplots_file = os.path.join(eval_dir, 'boxplots.eps')
    boxplots_pdf = os.path.join(eval_dir, 'boxplots.pdf')

    #fig, axes = plt.subplots(3, 1)
    fig, axes = plt.subplots(1, 1)
    axes.set(ylim=(-0.01, 1.01))
    fig.set_figheight(5)
    fig.set_figwidth(5)

    # Create an array with the colors you want to use
    colors = ["#FF0000", "#FFF200", "#0000FF"]# Set your custom color palette
    customPalette = sns.set_palette(sns.color_palette(colors))
    plt.rcParams['font.size'] = '14'
    
    #sns.color_palette("Spectral", as_cmap=True)
    sns.boxplot(x='Cardiac structure', y='3D Dice', data=df, ax=axes, boxprops=dict(alpha=.5))
    #sns.boxplot(x='Cardiac structure', y='HD',data=df, ax=axes[1], boxprops=dict(alpha=.5))
    #sns.boxplot(x='Cardiac structure', y='ASSD', data=df, ax=axes[2], boxprops=dict(alpha=.5))

    #plt.savefig(boxplots_file)
    plt.tight_layout()
    plt.savefig(boxplots_pdf)
    plt.close()

    return 0


def print_stats(df, eval_dir):

    out_file = os.path.join(eval_dir, 'summary_report.txt')

    with open(out_file, "w") as text_file:

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('Summary of geometric evaluation measures. \n')
        text_file.write('The following measures should be equivalent to those ')
        text_file.write('obtained from the online evaluation platform. \n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for struc_name in ['LV', 'RV', 'MYO']:

            text_file.write(struc_name)
            text_file.write('\n')

            for cardiac_phase in ['ED', 'ES']:

                text_file.write('    {}\n'.format(cardiac_phase))

                dat = df.loc[(df['phase'] == cardiac_phase) & (df['Cardiac structure'] == struc_name)]

                for measure_name in ['3D Dice', 'HD', 'ASSD']:

                    text_file.write('       {} -- mean (std): {:.3f} ({:.3f}) \n'.format(measure_name,
                                                                         np.mean(dat[measure_name]), np.std(dat[measure_name])))

                    ind_med = np.argsort(dat[measure_name]).iloc[len(dat[measure_name])//2]
                    text_file.write('             median {}: {:.3f} ({})\n'.format(measure_name,
                                                                dat[measure_name].iloc[ind_med], dat['filename'].iloc[ind_med]))

                    ind_worst = np.argsort(dat[measure_name]).iloc[0]
                    text_file.write('             worst {}: {:.3f} ({})\n'.format(measure_name,
                                                                dat[measure_name].iloc[ind_worst], dat['filename'].iloc[ind_worst]))

                    ind_best = np.argsort(dat[measure_name]).iloc[-1]
                    text_file.write('             best {}: {:.3f} ({})\n'.format(measure_name,
                                                                dat[measure_name].iloc[ind_best], dat['filename'].iloc[ind_best]))


        # text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        # text_file.write('Ejection fraction correlation between prediction and ground truth\n')
        # text_file.write('-------------------------------------------------------------------------------------\n\n')

        # for struc_name in ['LV', 'RV']:

        #     lv = df.loc[df['struc'] == struc_name]

        #     ED_vol = np.array(lv.loc[lv['phase'] == 'ED']['vol'])
        #     ES_vol = np.array(lv.loc[(lv['phase'] == 'ES')]['vol'])
        #     EF_pred = (ED_vol - ES_vol) / ED_vol

        #     ED_vol_gt = ED_vol - np.array(lv.loc[lv['phase'] == 'ED']['vol_err'])
        #     ES_vol_gt = ES_vol - np.array(lv.loc[(lv['phase'] == 'ES')]['vol_err'])

        #     EF_gt = (ED_vol_gt - ES_vol_gt) / ED_vol_gt

        #     LV_EF_corr = stats.pearsonr(EF_pred, EF_gt)
            # text_file.write('{}, EF corr: {}\n\n'.format(struc_name, LV_EF_corr[0]))

def load_nii(img_path):
    '''
    Shortcut to load a nifti file
    '''
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def main(path_gt, path_pred, eval_dir):
    """
    Calculate all sorts of geometric and clinical evaluation measures from the predicted segmentations.
    :param path_gt: path to ground truth segmentations
    :param path_pred: path to predicted segmentations
    :param eval_dir: directory where reports should be written
    :return:
    """

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    if os.path.isdir(path_gt) and os.path.isdir(path_pred):

        df = compute_metrics_on_directories_raw(path_gt, path_pred)

        print_stats(df, eval_dir)
        print_latex_tables(df, eval_dir)
        boxplot_metrics(df, eval_dir)

        logging.info('------------Average Dice Figures----------')
        logging.info('Dice 1 (LV): %f' % np.mean(df.loc[df['Cardiac structure'] == 'LV']['3D Dice']))
        logging.info('Dice 2 (RV): %f' % np.mean(df.loc[df['Cardiac structure'] == 'RV']['3D Dice']))
        logging.info('Dice 3 (MYO): %f' % np.mean(df.loc[df['Cardiac structure'] == 'MYO']['3D Dice']))
        logging.info('Mean Dice: %f' % np.mean(np.mean(df['3D Dice'])))
        logging.info('------------------------------------------')

    else:
        raise ValueError(
            "The paths given needs to be two directories or two files.")


if __name__ == "__main__":
    args = parser.get_arguments()
    #description="Script to compute ACDC challenge metrics.")
    # parser.add_argument("GT_IMG", type=str, help="Ground Truth image")
    # parser.add_argument("PRED_IMG", type=str, help="Predicted image")
    # parser.add_argument("EVAL_DIR", type=str, help="path to output directory", default='.')

    exp_name = args.exp_name + '/'
    #exp_name = '2021-10-11-acdc_test_ep49_stdAug_ImageNetBackbone_lr_0.05_budget_1904_seed_234/'
    path_eval = '/mnt/qb/baumgartner/cschmidt77_data/' + exp_name
    path_gt = path_eval + 'ground_truth/'
    path_pred = path_eval + 'prediction/'

    print("path_eval: ", path_eval)
    print("path_gt: ", path_gt)
    print("path_pred: ", path_pred)
    main(path_gt, path_pred, path_eval)
# exp-name: /mnt/qb/baumgartner/cschmidt77_data/_FINAL_ACDC_exp4_3D_baselines_ralis/2021-11-03-acdc_ImageNetBackbone_baseline_entropy_budget_1184_seed_77


#singularity exec --nv --bind /mnt/qb/baumgartner ralis.sif python3 -u devel/ralis/metrics_acdc.py --exp-name '_FINAL_ACDC_exp4_3D_baselines_ralis/2021-11-03-acdc_ImageNetBackbone_baseline_entropy_budget_1184_seed_77' --checkpointer --data-path '/mnt/qb/baumgartner/cschmidt77_data/'  --dataset 'acdc' --al-algorithm 'ralis'
# --ckpt-path '/mnt/qb/baumgartner/cschmidt77_data/exp2_acdc_supervised_dice' 

#best RALIS for 20% 
#/mnt/qb/baumgartner/cschmidt77_data/_FINAL_ACDC_exp4_3D_baselines_ralis/2021-11-04-acdc_ralis_ImageNetBackbone_budget_2384_seed_123
