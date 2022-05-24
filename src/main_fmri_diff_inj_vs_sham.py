from tqdm import tqdm
from glob import glob
import nilearn.image as ni
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
from brainsync import normalizeData, brainSync, groupBrainSync
from logging import error
from scipy import io as spio
from scipy.stats import ranksums, ttest_ind, ttest_rel
import os
from statsmodels.stats.power import TTestIndPower
import csv

#from surfproc import patch_color_attrib, smooth_surf_function

#from dfsio import readdfs, writedfs

# correct if the population S.D. is expected to be equal for the two groups.


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


def save_roiwise_fmri(data, origfmrifile, outfmrifile, labelsfile, label_ids):

    labels = ni.load_img(labelsfile).get_fdata()
    fmri = ni.load_img(origfmrifile).get_fdata()
    num_time = fmri.shape[3]
    num_rois = len(label_ids)
    rtseries = np.zeros(fmri.shape)

    # Copy the mean synced time series to each ROI
    for i, id in enumerate(label_ids):
        rtseries[labels == id, :] = data[:, i]

    nii = ni.new_img_like(origfmrifile, rtseries)

    nii.to_filename(outfmrifile)


def get_roiwise_fmri(fmri, labels, label_ids):

    labels = ni.load_img(labels).get_fdata()
    fmri = ni.load_img(fmri).get_fdata()
    num_time = fmri.shape[3]
    num_rois = len(label_ids)
    rtseries = np.zeros((num_time, num_rois))

    for i, id in enumerate(label_ids):
        rtseries[:, i] = np.mean(fmri[labels == id, :], axis=0)

    rtseries_norm, _, _ = normalizeData(rtseries)

    return rtseries_norm, rtseries


def get_fmri_diff_tpts(dir_7d, dir_28d):
    flist = glob(dir_7d + '/at*.nii.gz')
    label_ids = np.arange(83, dtype=np.int16)
    num_time = 450

    # remove WM label from connectivity analysis
    label_ids = np.setdiff1d(label_ids, 0)
    num_rois = label_ids.shape[0]
    num_sub = len(flist)


# Get a list of subjects
    sublist = list()
    for f in flist:
        pth, fname = os.path.split(f)
        sublist.append(fname[6:-7])  # 26

    fmri_roiwise_7d_all = np.zeros([num_time, num_rois, num_sub])
    fmri_roiwise_28d_all = np.zeros([num_time, num_rois, num_sub])
    fmri_roiwise_28d_synced_all = np.zeros([num_time, num_rois, num_sub])
    fmri_tdiff_all = np.zeros([num_rois, num_sub])

    for i, sub in enumerate(tqdm(sublist)):

        # Add extension for 7d
        sub_7d = sub

        sub_28d = sub_7d.replace('7d_rsfmri', '28d_rsfmri')
        sub_28d = sub_28d.replace('std_07', 'std_28')

        f = glob(dir_28d + '/' + sub[:2] + '*.nii.gz')
        d, s = os.path.split(f[0])
        if len(f) != 1:
            error('error in 28th day timepoint files for ' + s)

        sub_28d = s[:-7]

        fmri_7d = os.path.join(dir_7d, sub_7d + '.nii.gz')
        labels_7d = os.path.join(dir_7d, 'atlas_' + sub_7d + '.nii.gz')

        fmri_28d = os.path.join(dir_28d, sub_28d + '.nii.gz')
        labels_28d = os.path.join(
            dir_28d, 'atlas_' + sub_28d + '.nii.gz')

        fmri_roiwise_7d_all[:, :, i], _ = get_roiwise_fmri(
            fmri_7d, labels_7d, label_ids)
        fmri_roiwise_28d_all[:, :, i], _ = get_roiwise_fmri(
            fmri_28d, labels_28d, label_ids)

        d, _ = brainSync(
            fmri_roiwise_7d_all[:, :, i], fmri_roiwise_28d_all[:, :, i])

        fmri_roiwise_28d_synced_all[:, :, i] = d

        fmri_tdiff_all[:, i] = np.linalg.norm(
            fmri_roiwise_7d_all[:, :, i] - d, axis=0)

    return fmri_tdiff_all, fmri_roiwise_28d_synced_all, fmri_roiwise_7d_all, fmri_roiwise_28d_all


def plot_atlas_pval(atlas_fname, roi_ids, pval, out_fname, alpha=0.05):

    atlas = ni.load_img(atlas_fname)
    atlas_img = atlas.get_fdata()

    img = np.ones(atlas.shape)

    for i, roi in enumerate(roi_ids):
        img[atlas_img == roi] = pval[i]

    pval_vol = ni.new_img_like(atlas, img)

    pval_vol.to_filename(out_fname + '.nii.gz')

    img[img > alpha] = alpha
    pval_vol = ni.new_img_like(atlas, alpha - img)

    plotting.plot_stat_map(bg_img=atlas, stat_map_img=pval_vol, vmax=alpha, threshold=0.0, output_file=out_fname + '.png',
                           draw_cross=False, annotate=True, display_mode="ortho", cut_coords=[(85-68)*1.25, (111-90)*1.25, (54-51)*1.25])
    plt.show()


def plot_atlas_var(atlas_fname, roi_ids, roi_var, out_fname, vmax=0.001, vmin=0):
    """ Plot variance computed for each roi """

    atlas = ni.load_img(atlas_fname)
    atlas_img = atlas.get_fdata()

    img = np.zeros(atlas.shape)

    for i, roi in enumerate(roi_ids):
        img[atlas_img == roi] = roi_var[i]

    val_vol = ni.new_img_like(atlas, img)

    val_vol.to_filename(out_fname + '.nii.gz')
    val_vol = ni.new_img_like(atlas, img)

    # plot var
    plotting.plot_img(img=val_vol, threshold=vmin, output_file=out_fname + '.png', draw_cross=False, annotate=True, black_bg=True,
                      display_mode="ortho", cut_coords=[(85-68)*1.25, (111-90)*1.25, (54-51)*1.25], vmax=vmax, vmin=vmin, colorbar=True, cmap='hot')
    plt.show()


def fmri_sync(fmri, Os):
    """Sync gmri data using given Os"""
    for j in range(fmri.shape[2]):
        fmri[:, :, j] = np.dot(Os[:, :, j], fmri[:, :, j])

    return fmri


if __name__ == "__main__":

    dir_7d = '/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/shm_07d/'
    dir_28d = '/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/shm_28d/'
    # dir with synced nifti files for shm group
    dir_28d_synced = '/home/ajoshi/Desktop/shm_28d_synced/'
    atlas_fname = '/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/01_study_specific_atlas_relabel.nii.gz'
##
    fmri_tdiff_shm_all, fmri_shm_28d_synced_all, fmri_shm_7d_all, fmri_shm_28d_all = get_fmri_diff_tpts(
        dir_7d, dir_28d)
    np.savez('shm.npz', fmri_tdiff_inj_all=fmri_tdiff_shm_all)
    # saved as time x roi x subject
    np.savez('shm_synced_28d_to_7d.npz', fmri_shm_28d_synced_all=fmri_shm_28d_synced_all)

    dir_7d = '/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/inj_07d/'
    dir_28d = '/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/inj_28d/'
    # dir with synced nifti files for inj group
    dir_28d_synced = '/home/ajoshi/Desktop/inj_28d_synced/'

    fmri_tdiff_inj_all, fmri_inj_28d_synced_all, fmri_inj_7d_all, fmri_inj_28d_all = get_fmri_diff_tpts(
        dir_7d, dir_28d)

    np.savez('inj.npz', fmri_tdiff_inj_all=fmri_tdiff_inj_all)
    # saved as time x roi x subject
    np.savez('inj_synced_28d_to_7d.npz', fmri_inj_28d_synced_all=fmri_inj_28d_synced_all)

    num_rois = fmri_tdiff_inj_all.shape[0]
    pval2 = np.zeros(num_rois)
    pval = np.zeros(num_rois)
    pval_opp = np.zeros(num_rois)

    for r in tqdm(range(num_rois)):
        _, pval2[r] = ttest_ind(
            fmri_tdiff_inj_all[r, ], fmri_tdiff_shm_all[r, ], alternative='two-sided', equal_var=False)
        _, pval[r] = ttest_ind(
            fmri_tdiff_inj_all[r, ], fmri_tdiff_shm_all[r, ], alternative='less', equal_var=False)
        _, pval_opp[r] = ttest_ind(
            fmri_tdiff_inj_all[r, ], fmri_tdiff_shm_all[r, ], alternative='greater', equal_var=False)

    np.savez('pval.npz', pval2=pval2, pval=pval, pval_opp=pval_opp)
    print(np.stack((pval, pval2, pval_opp)).T)
##
    plot_atlas_pval(atlas_fname, np.arange(1, num_rois+1),
                    pval, out_fname='pval_7d_28d', alpha=0.25)
    plot_atlas_pval(atlas_fname, np.arange(1, num_rois+1),
                    pval2, out_fname='pval2_7d_28d', alpha=0.25)
    plot_atlas_pval(atlas_fname, np.arange(1, num_rois+1),
                    pval_opp, out_fname='pval_opp_7d_28d', alpha=0.25)

##
    # Calculate variance of 7d sham
    a, Os, Costdif, TotalError = groupBrainSync(fmri_shm_7d_all)
    fmri_shm_7d_all_synced = fmri_sync(fmri_shm_7d_all, Os)

    np.savez('shm_7d_grp_synced.npz',fmri_shm_7d_all_synced=fmri_shm_7d_all_synced)

    fmri_atlas_7d_shm = np.mean(fmri_shm_7d_all_synced, axis=2)
    var_7d_shm = np.mean(
        (fmri_shm_7d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_fname, np.arange(1, num_rois+1),
                   var_7d_shm, out_fname='var_7d_shm', vmax=0.0006, vmin=0.0004)
    dist2atlas_7d_shm = np.sum(
        (fmri_shm_7d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0))
##
    # Calculate variance of 28d sham
    a, Os, Costdif, TotalError = groupBrainSync(fmri_shm_28d_all)
    fmri_shm_28d_all_synced = fmri_sync(fmri_shm_28d_all, Os)

    np.savez('shm_28d_grp_synced.npz',fmri_shm_28d_all_synced=fmri_shm_28d_all_synced)

    fmri_atlas = np.mean(fmri_shm_28d_all_synced, axis=2)
    var_28d_shm = np.mean(
        (fmri_shm_28d_all_synced - fmri_atlas[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_fname, np.arange(1, num_rois+1),
                   var_28d_shm, out_fname='var_28d_shm', vmax=0.0006, vmin=0.0004)

##
    # Calculate variance of 7d inj
    a, Os, Costdif, TotalError = groupBrainSync(fmri_inj_7d_all)
    fmri_inj_7d_all_synced = fmri_sync(fmri_inj_7d_all, Os)

    np.savez('inj_7d_grp_synced.npz',fmri_inj_7d_all_synced=fmri_inj_7d_all_synced)

    fmri_atlas = np.mean(fmri_inj_7d_all_synced, axis=2)
    var_7d_inj = np.mean(
        (fmri_inj_7d_all_synced - fmri_atlas[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_fname, np.arange(1, num_rois+1),
                   var_7d_inj, out_fname='var_7d_inj', vmax=0.0006, vmin=0.0004)

    # Calculate variance of 28d inj
    a, Os, Costdif, TotalError = groupBrainSync(fmri_inj_28d_all)
    fmri_inj_28d_all_synced = fmri_sync(fmri_inj_28d_all, Os)

    np.savez('inj_28d_grp_synced.npz',fmri_inj_28d_all_synced=fmri_inj_28d_all_synced)

    fmri_atlas = np.mean(fmri_inj_28d_all_synced, axis=2)
    var_28d_inj = np.mean(
        (fmri_inj_28d_all_synced - fmri_atlas[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_fname, np.arange(1, num_rois+1),
                   var_28d_inj, out_fname='var_28d_inj', vmax=0.0006, vmin=0.0004)

    # Calculate variance of 28d shm wrt 7d shm grp atlas
    num_sub = fmri_shm_28d_all.shape[2]
    fmri_shm_28d_all_synced = np.zeros(fmri_shm_28d_all.shape)

    for ind in range(num_sub):
        fmri_shm_28d_all_synced[:, :, ind], _ = brainSync(
            fmri_atlas_7d_shm, fmri_shm_28d_all[:, :, ind])

    np.savez('shm_28d_synced_to_7d_shm_atlas.npz', fmri_shm_28d_all_synced=fmri_shm_28d_all_synced)

    dist2atlas_28d_shm = np.sum(
        (fmri_shm_28d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0))
    var_28d_shm = np.mean(
        (fmri_shm_28d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_fname, np.arange(1, num_rois+1),
                   var_28d_shm, out_fname='var_28d_shm_7d_shm', vmax=0.0006, vmin=0.0004)

    # Calculate variance of 7d inj wrt 7d shm grp atlas
    num_sub = fmri_inj_7d_all.shape[2]
    fmri_inj_7d_all_synced = np.zeros(fmri_inj_7d_all.shape)

    for ind in range(num_sub):
        fmri_inj_7d_all_synced[:, :, ind], _ = brainSync(
            fmri_atlas_7d_shm, fmri_inj_7d_all[:, :, ind])

    np.savez('inj_7d_synced_to_7d_shm_atlas.npz', fmri_inj_7d_all_synced=fmri_inj_7d_all_synced)

    dist2atlas_7d_inj = np.sum(
        (fmri_inj_7d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0))
    var_7d_inj = np.mean(
        (fmri_inj_7d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_fname, np.arange(1, num_rois+1),
                   var_7d_inj, out_fname='var_7d_inj_7d_shm', vmax=0.0006, vmin=0.0004)

    # Calculate variance of 28d inj wrt 7d shm grp atlas
    num_sub = fmri_inj_28d_all.shape[2]
    fmri_inj_28d_all_synced = np.zeros(fmri_inj_28d_all.shape)

    for ind in range(num_sub):
        fmri_inj_28d_all_synced[:, :, ind], _ = brainSync(
            fmri_atlas_7d_shm, fmri_inj_28d_all[:, :, ind])

    np.savez('inj_28d_synced_to_7d_shm_atlas.npz', fmri_inj_28d_all_synced=fmri_inj_28d_all_synced)

    dist2atlas_28d_inj = np.sum(
        (fmri_inj_28d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0))
    var_28d_inj = np.mean(
        (fmri_inj_28d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_fname, np.arange(1, num_rois+1),
                   var_28d_inj, out_fname='var_28d_inj_7d_shm', vmax=0.0006, vmin=0.0004)


##
# Which ROIs are affected in TBI: 7d inj vs 7d non-injury
# Which ROIs get better: 7d inj vs 28d injury
# Which ROIs get worse in TBI: 7d inj vs 28d injury

    pval = np.zeros(num_rois)
    pval2 = np.zeros(num_rois)
    pval3 = np.zeros(num_rois)

    for r in tqdm(range(num_rois)):
        _, pval[r] = ttest_ind(
            dist2atlas_7d_inj[r, ], dist2atlas_7d_shm[r, ], alternative='greater', equal_var=False)
        _, pval2[r] = ttest_rel(dist2atlas_7d_inj[r, ],
                                dist2atlas_28d_inj[r, ], alternative='greater')
        _, pval3[r] = ttest_rel(dist2atlas_7d_inj[r, ],
                                dist2atlas_28d_inj[r, ], alternative='less')

    plot_atlas_pval(atlas_fname, np.arange(1, num_rois+1),
                    pval, out_fname='rois_affected', alpha=0.05)
    plot_atlas_pval(atlas_fname, np.arange(1, num_rois+1),
                    pval2, out_fname='rois_get_better', alpha=0.05)
    plot_atlas_pval(atlas_fname, np.arange(1, num_rois+1),
                    pval3, out_fname='rois_get_worse', alpha=0.05)

    # Write the p values to csv file
    fieldnames = ["ROI ID", "pval_affected",
                  "pval_get_better", "pval_get_worse"]

    roiIDs = np.arange(1, 83)
    with open('rois_affected_rois_get_better_rois_get_worse_pvalues.csv', 'w') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
        writer.writeheader()
        for i, roiid in enumerate(roiIDs):
            writer.writerow({'ROI ID': roiIDs[i], "pval_affected": pval[i],
                            "pval_get_better": pval2[i], "pval_get_worse": pval3[i]})


# Cohen's d
    cohen_d1 = np.zeros(num_rois)
    cohen_d2 = np.zeros(num_rois)
    cohen_d3 = np.zeros(num_rois)
    tt_power = np.zeros(num_rois)

    for r in tqdm(range(num_rois)):
        cohen_d1[r] = cohen_d(dist2atlas_7d_inj[r, ], dist2atlas_7d_shm[r, ])
        cohen_d2[r] = cohen_d(dist2atlas_7d_inj[r, ], dist2atlas_28d_inj[r, ])
        cohen_d3[r] = cohen_d(dist2atlas_7d_inj[r, ], dist2atlas_28d_inj[r, ])
        analysis = TTestIndPower()
        tt_power[r] = analysis.power(cohen_d1[r], nobs1=len(dist2atlas_7d_inj[r, ]), alpha=0.05, ratio=len(
            dist2atlas_7d_shm[r, ])/len(dist2atlas_7d_shm[r, ]))

    print('Please Note that colorbars should go from 0 to 2, \
        but due to limitation of the nilearn functions, the data is scaled by a factor of 2')
    plot_atlas_pval(atlas_fname, np.arange(1, num_rois+1),
                    (1-tt_power), out_fname='rois_affected_tt_power', alpha=1)
    plot_atlas_pval(atlas_fname, np.arange(1, num_rois+1),
                    (2-np.abs(cohen_d1))/2, out_fname='rois_affected_cohen_d', alpha=1)
    plot_atlas_pval(atlas_fname, np.arange(1, num_rois+1),
                    (2-np.abs(cohen_d2))/2, out_fname='rois_get_better_cohen_d', alpha=1)
    plot_atlas_pval(atlas_fname, np.arange(1, num_rois+1),
                    (2-np.abs(cohen_d3))/2, out_fname='rois_get_worse_cohen_d', alpha=1)

    input('press any key')
