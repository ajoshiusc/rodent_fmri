#!/usr/bin/env python
from tqdm import tqdm
from glob import glob
import nilearn.image as ni
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
from brainsync import normalizeData, brainSync, groupBrainSync
from scipy import io as spio
from scipy.stats import mannwhitneyu, wilcoxon
from scipy.stats import rankdata
import os
import csv
import argparse

#from surfproc import patch_color_attrib, smooth_surf_function

#from dfsio import readdfs, writedfs

# correct if the population S.D. is expected to be equal for the two groups.


def cliffs_delta(x, y):
    m, n = len(x), len(y)
    mat = np.sign(np.subtract.outer(x, y))
    return np.sum(mat) / (m * n)

def wilcoxon_effect_size(x, y):
    d = np.array(x) - np.array(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    pos_ranks = np.sum(ranks[d > 0])
    neg_ranks = np.sum(ranks[d < 0])
    total_ranks = pos_ranks + neg_ranks
    return (pos_ranks - neg_ranks) / total_ranks

def nonparametric_power_ind(x, y, alpha=0.05, n_sim=1000):
    n_x, n_y = len(x), len(y)
    sig_count = 0
    for _ in range(n_sim):
        x_sim = np.random.choice(x, size=n_x, replace=True)
        y_sim = np.random.choice(y, size=n_y, replace=True)
        try:
            # Use alternative='two-sided' to match the test used for pval
            _, p = mannwhitneyu(x_sim, y_sim, alternative='two-sided')
            if p < alpha:
                sig_count += 1
        except ValueError:
            pass
    return sig_count / n_sim


def save_roiwise_fmri(data, origfmrifile, outfmrifile, labelsfile, label_ids):

    labels = ni.load_img(labelsfile).get_fdata()
    fmri = ni.load_img(origfmrifile).get_fdata()
    rtseries = np.zeros(fmri.shape)

    # Copy the mean synced time series to each ROI
    for i, roi_id in enumerate(label_ids):
        rtseries[labels == roi_id, :] = data[:, i]

    nii = ni.new_img_like(origfmrifile, rtseries)

    nii.to_filename(outfmrifile)


def get_roiwise_fmri(fmri, labels, label_ids):

    labels = ni.load_img(labels).get_fdata()
    fmri = ni.load_img(fmri).get_fdata()
    num_time = fmri.shape[3]
    num_rois = len(label_ids)
    rtseries = np.zeros((num_time, num_rois))

    for i, roi_id in enumerate(label_ids):
        rtseries[:, i] = np.mean(fmri[labels == roi_id, :], axis=0)

    rtseries_norm, _, _ = normalizeData(rtseries)

    return rtseries_norm, rtseries


def get_fmri_diff_tpts(dir_7d, dir_28d):
    flist = glob(dir_7d + '/at*.nii.gz')
    print(flist)
    label_ids = np.arange(83, dtype=np.int16)
    num_time = 450

    # remove WM label from connectivity analysis
    label_ids = np.setdiff1d(label_ids, 0)
    num_rois = label_ids.shape[0]
    num_sub = len(flist)


# Get a list of subjects
    sublist = list()
    for f in flist:
        _, fname = os.path.split(f)
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
        if len(f) != 1:
            raise FileNotFoundError(
                f'Expected exactly one 28d file for subject prefix {sub[:2]}, found {len(f)}')

        _, s = os.path.split(f[0])

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



def plot_atlas_pval(atlas_image, atlas_labels, roi_ids, pval, out_fname, alpha=0.05):

    atlas = ni.load_img(atlas_labels)
    atlas_img = atlas.get_fdata()

    img = np.ones(atlas.shape)

    for i, roi in enumerate(roi_ids):
        img[atlas_img == roi] = pval[i]

    pval_vol = ni.new_img_like(atlas, img)

    pval_vol.to_filename(out_fname + "_brainsync.nii.gz")

    img[img > alpha] = alpha
    pval_vol = ni.new_img_like(atlas, alpha - img)

    # plotting.plot_stat_map(bg_img=atlas_image, stat_map_img=pval_vol, vmax=alpha, threshold=0.0, output_file=out_fname + '_w.png',
    #                       draw_cross=False, annotate=True, display_mode="ortho", cut_coords=[(85-68)*1.25, (111-90)*1.25, (54-51)*1.25])
    plotting.plot_stat_map(
        bg_img=atlas_image,
        stat_map_img=pval_vol,
        vmax=alpha,
        threshold=0.0,
        output_file=out_fname + "_w_brainsync.png",
        draw_cross=False,
        annotate=False,
        display_mode="y",
        cut_coords=[(111 - 90) * 1.25],
        cmap="hot",
        colorbar=False,
        #vmin=0,
    )

    plt.show()



def plot_atlas_var(atlas_image, atlas_labels, roi_ids, roi_var, out_fname):
    """Plot variance computed for each roi"""

    atlas = ni.load_img(atlas_labels)
    atlas_img = atlas.get_fdata()

    img = np.zeros(atlas.shape)

    for i, roi in enumerate(roi_ids):
        img[atlas_img == roi] = roi_var[i]

    val_vol = ni.new_img_like(atlas, img)

    val_vol.to_filename(out_fname + ".nii.gz")
    val_vol = ni.new_img_like(atlas, img)

    # plot var

    # plotting.plot_stat_map(bg_img=atlas_image, stat_map_img=val_vol, threshold=0.0, output_file=out_fname + '_w.png', draw_cross=False,
    #                       annotate=True, display_mode="ortho", cut_coords=[(85-68)*1.25, (111-90)*1.25, (54-51)*1.25], vmax=0.001)
    plotting.plot_stat_map(
        bg_img=atlas_image,
        stat_map_img=val_vol,
        threshold=0.0,
        output_file=out_fname + "_w_brainsync.png",
        draw_cross=False,
        annotate=False,
        display_mode="y",
        cut_coords=[(111 - 90) * 1.25],
        vmax=0.001,
        vmin=0,
        colorbar=False,
        cmap="hot",
    )

    plt.show()


def fmri_sync(fmri, Os):
    """Sync gmri data using given Os"""
    fmri_synced = np.zeros_like(fmri)
    for j in range(fmri.shape[2]):
        fmri_synced[:, :, j] = np.dot(Os[:, :, j], fmri[:, :, j])

    return fmri_synced


if __name__ == "__main__":
    dstdir='/home/ajoshi/Desktop/rod_tbi/nonparametric_brainsync_results'
    srcdir='/deneb_disk'
    parser = argparse.ArgumentParser(
                    prog='main_fmri_diff_inj_vs_sham_nonparametric.py',
                    description='comparison of subjects in rodent fMRI study using nonparametric tests and brain sync')
    parser.add_argument('--srcdir','-s', default=srcdir, help='source directory for data')
    parser.add_argument('--dstdir','-d', default=dstdir, help='output directory')
    args = parser.parse_args()
    dstdir=os.path.realpath(args.dstdir)
    srcdir=os.path.realpath(args.srcdir)
    os.makedirs(dstdir, exist_ok=True)
    dir_7d = f'{srcdir}/ucla_mouse_injury/ucla_injury_rats/shm_07d/'
    dir_28d = f'{srcdir}/ucla_mouse_injury/ucla_injury_rats/shm_28d/'
    # dir with synced nifti files for shm group
    dir_28d_synced = f'{dstdir}/shm_28d_synced/'
    atlas_labels = f'{srcdir}/ucla_mouse_injury/ucla_injury_rats/01_study_specific_atlas_relabel.nii.gz'
    atlas_image = f'{srcdir}/ucla_mouse_injury/ucla_injury_rats/brain.nii.gz'
##  
    fmri_tdiff_shm_all, fmri_shm_28d_synced_all, fmri_shm_7d_all, fmri_shm_28d_all = get_fmri_diff_tpts(
        dir_7d, dir_28d)
    np.savez(f'{dstdir}/shm_nonparametric.npz', fmri_tdiff_shm_all=fmri_tdiff_shm_all)
    # saved as time x roi x subject
    spio.savemat(f'{dstdir}/shm_synced_28d_to_7d_nonparametric.mat', {'fmri_shm_28d_synced_all':fmri_shm_28d_synced_all})

    dir_7d = f'{srcdir}/ucla_mouse_injury/ucla_injury_rats/inj_07d/'
    dir_28d = f'{srcdir}/ucla_mouse_injury/ucla_injury_rats/inj_28d/'
    # dir with synced nifti files for inj group
    dir_28d_synced = f'{dstdir}/inj_28d_synced/'

    fmri_tdiff_inj_all, fmri_inj_28d_synced_all, fmri_inj_7d_all, fmri_inj_28d_all = get_fmri_diff_tpts(
        dir_7d, dir_28d)

    np.savez(f'{dstdir}/inj_nonparametric.npz', fmri_tdiff_inj_all=fmri_tdiff_inj_all)
    # saved as time x roi x subject
    spio.savemat(f'{dstdir}/inj_synced_28d_to_7d_nonparametric.mat', {'fmri_inj_28d_synced_all':fmri_inj_28d_synced_all})

    num_rois = fmri_tdiff_inj_all.shape[0]
    pval2 = np.zeros(num_rois)
    pval = np.zeros(num_rois)
    pval_opp = np.zeros(num_rois)

    for r in tqdm(range(num_rois)):
        _, pval2[r] = mannwhitneyu(
            fmri_tdiff_inj_all[r, ], fmri_tdiff_shm_all[r, ], alternative='two-sided')
        _, pval[r] = mannwhitneyu(
            fmri_tdiff_inj_all[r, ], fmri_tdiff_shm_all[r, ], alternative='less')
        _, pval_opp[r] = mannwhitneyu(
            fmri_tdiff_inj_all[r, ], fmri_tdiff_shm_all[r, ], alternative='greater')

    from statsmodels.stats.multitest import fdrcorrection
    _, pval_fdr = fdrcorrection(pval, alpha=0.05)
    _, pval2_fdr = fdrcorrection(pval2, alpha=0.05)
    _, pval_opp_fdr = fdrcorrection(pval_opp, alpha=0.05)

    np.savez(f'{dstdir}/pval_nonparametric.npz', pval2=pval2_fdr, pval=pval_fdr, pval_opp=pval_opp_fdr)
    print(np.stack((pval_fdr, pval2_fdr, pval_opp_fdr)).T)
##
    plot_atlas_pval(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                    pval_fdr, out_fname=f'{dstdir}/pval_7d_28d_nonparametric', alpha=0.05)
    plot_atlas_pval(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                    pval2_fdr, out_fname=f'{dstdir}/pval2_7d_28d_nonparametric', alpha=0.05)
    plot_atlas_pval(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                    pval_opp_fdr, out_fname=f'{dstdir}/pval_opp_7d_28d_nonparametric', alpha=0.05)

##
    # Calculate variance of 7d sham
    a, Os, Costdif, TotalError = groupBrainSync(fmri_shm_7d_all)
    fmri_shm_7d_all_synced = fmri_sync(fmri_shm_7d_all, Os)

    spio.savemat(f'{dstdir}/shm_7d_grp_synced.mat',{'fmri_shm_7d_all_synced':fmri_shm_7d_all_synced})

    fmri_atlas_7d_shm = np.mean(fmri_shm_7d_all_synced, axis=2)
    var_7d_shm = np.mean(
        (fmri_shm_7d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                   var_7d_shm, out_fname=f'{dstdir}/var_7d_shm_nonparametric')#, vmax=0.0006, vmin=0.0004)
    dist2atlas_7d_shm = np.sum(
        (fmri_shm_7d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0))
##
    # Calculate variance of 28d sham
    a, Os, Costdif, TotalError = groupBrainSync(fmri_shm_28d_all)
    fmri_shm_28d_all_synced = fmri_sync(fmri_shm_28d_all, Os)

    spio.savemat(f'{dstdir}/shm_28d_grp_synced_nonparametric.mat',{'fmri_shm_28d_all_synced':fmri_shm_28d_all_synced})

    fmri_atlas = np.mean(fmri_shm_28d_all_synced, axis=2)
    var_28d_shm = np.mean(
        (fmri_shm_28d_all_synced - fmri_atlas[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                   var_28d_shm, out_fname=f'{dstdir}/var_28d_shm_nonparametric')#, vmax=0.0006, vmin=0.0004)

##
    # Calculate variance of 7d inj
    a, Os, Costdif, TotalError = groupBrainSync(fmri_inj_7d_all)
    fmri_inj_7d_all_synced = fmri_sync(fmri_inj_7d_all, Os)

    spio.savemat(f'{dstdir}/inj_7d_grp_synced_nonparametric.mat',{'fmri_inj_7d_all_synced':fmri_inj_7d_all_synced})

    fmri_atlas = np.mean(fmri_inj_7d_all_synced, axis=2)
    var_7d_inj = np.mean(
        (fmri_inj_7d_all_synced - fmri_atlas[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                   var_7d_inj, out_fname=f'{dstdir}/var_7d_inj_nonparametric')#, vmax=0.0006, vmin=0.0004)

    # Calculate variance of 28d inj
    a, Os, Costdif, TotalError = groupBrainSync(fmri_inj_28d_all)
    fmri_inj_28d_all_synced = fmri_sync(fmri_inj_28d_all, Os)

    spio.savemat(f'{dstdir}/inj_28d_grp_synced_nonparametric.mat',{'fmri_inj_28d_all_synced':fmri_inj_28d_all_synced})

    fmri_atlas = np.mean(fmri_inj_28d_all_synced, axis=2)
    var_28d_inj = np.mean(
        (fmri_inj_28d_all_synced - fmri_atlas[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                   var_28d_inj, out_fname=f'{dstdir}/var_28d_inj_nonparametric')#, vmax=0.0006, vmin=0.0004)

    # Calculate variance of 28d shm wrt 7d shm grp atlas
    num_sub = fmri_shm_28d_all.shape[2]
    fmri_shm_28d_all_synced = np.zeros(fmri_shm_28d_all.shape)

    for ind in range(num_sub):
        fmri_shm_28d_all_synced[:, :, ind], _ = brainSync(
            fmri_atlas_7d_shm, fmri_shm_28d_all[:, :, ind])

    spio.savemat(f'{dstdir}/shm_28d_synced_to_7d_shm_atlas_nonparametric.mat', {'fmri_shm_28d_all_synced':fmri_shm_28d_all_synced})

    dist2atlas_28d_shm = np.sum(
        (fmri_shm_28d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0))
    var_28d_shm = np.mean(
        (fmri_shm_28d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                   var_28d_shm, out_fname=f'{dstdir}/var_28d_shm_7d_shm_nonparametric')#, vmax=0.0006, vmin=0.0004)

    # Calculate variance of 7d inj wrt 7d shm grp atlas
    num_sub = fmri_inj_7d_all.shape[2]
    fmri_inj_7d_all_synced = np.zeros(fmri_inj_7d_all.shape)

    for ind in range(num_sub):
        fmri_inj_7d_all_synced[:, :, ind], _ = brainSync(
            fmri_atlas_7d_shm, fmri_inj_7d_all[:, :, ind])

    spio.savemat(f'{dstdir}/inj_7d_synced_to_7d_shm_atlas_nonparametric.mat', {'fmri_inj_7d_all_synced':fmri_inj_7d_all_synced})

    dist2atlas_7d_inj = np.sum(
        (fmri_inj_7d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0))
    var_7d_inj = np.mean(
        (fmri_inj_7d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                   var_7d_inj, out_fname=f'{dstdir}/var_7d_inj_7d_shm_nonparametric')#, vmax=0.0006, vmin=0.0004)

    # Calculate variance of 28d inj wrt 7d shm grp atlas
    num_sub = fmri_inj_28d_all.shape[2]
    fmri_inj_28d_all_synced = np.zeros(fmri_inj_28d_all.shape)

    for ind in range(num_sub):
        fmri_inj_28d_all_synced[:, :, ind], _ = brainSync(
            fmri_atlas_7d_shm, fmri_inj_28d_all[:, :, ind])

    spio.savemat(f'{dstdir}/inj_28d_synced_to_7d_shm_atlas_nonparametric.mat', {'fmri_inj_28d_all_synced':fmri_inj_28d_all_synced})

    dist2atlas_28d_inj = np.sum(
        (fmri_inj_28d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0))
    var_28d_inj = np.mean(
        (fmri_inj_28d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                   var_28d_inj, out_fname=f'{dstdir}/var_28d_inj_7d_shm_nonparametric')#, vmax=0.0006, vmin=0.0004)


##
# Which ROIs are affected in TBI: 7d inj vs 7d non-injury
# Which ROIs get better: 7d inj vs 28d injury
# Which ROIs get worse in TBI: 7d inj vs 28d injury

    from statsmodels.stats.multitest import fdrcorrection
    
    pval = np.zeros(num_rois)
    pval2 = np.zeros(num_rois)
    pval3 = np.zeros(num_rois)

    for r in tqdm(range(num_rois)):
        _, pval[r] = mannwhitneyu(
            dist2atlas_7d_inj[r, ], dist2atlas_7d_shm[r, ], alternative='two-sided')
        _, pval2[r] = wilcoxon(dist2atlas_7d_inj[r, ],
                                dist2atlas_28d_inj[r, ], alternative='greater')
        _, pval3[r] = wilcoxon(dist2atlas_7d_inj[r, ],
                                dist2atlas_28d_inj[r, ], alternative='less')

    # Apply FDR correction
    _, pval_fdr = fdrcorrection(pval, alpha=0.05)
    _, pval2_fdr = fdrcorrection(pval2, alpha=0.05)
    _, pval3_fdr = fdrcorrection(pval3, alpha=0.05)

    plot_atlas_pval(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                    pval_fdr, out_fname=f'{dstdir}/rois_affected_nonparametric', alpha=0.05)
    plot_atlas_pval(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                    pval2_fdr, out_fname=f'{dstdir}/rois_get_better_nonparametric', alpha=0.05)
    plot_atlas_pval(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                    pval3_fdr, out_fname=f'{dstdir}/rois_get_worse_nonparametric', alpha=0.05)

    # Write the p values to csv file
    fieldnames = ["ROI ID", "pval_affected",
                  "pval_get_better", "pval_get_worse"]

    roiIDs = np.arange(1, 83)
    with open(f'{dstdir}/rois_affected_rois_get_better_rois_get_worse_pvalues_nonparametric.csv', 'w', encoding='utf-8', newline='') as outcsv:
        writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
        writer.writeheader()
        for i, roiid in enumerate(roiIDs):
            writer.writerow({'ROI ID': roiIDs[i], "pval_affected": pval_fdr[i],
                            "pval_get_better": pval2_fdr[i], "pval_get_worse": pval3_fdr[i]})


# Nonparametric effect sizes
    effect_size1 = np.zeros(num_rois)
    effect_size2 = np.zeros(num_rois)
    effect_size3 = np.zeros(num_rois)
    np_power = np.zeros(num_rois)

    for r in tqdm(range(num_rois)):
        effect_size1[r] = cliffs_delta(dist2atlas_7d_inj[r, ], dist2atlas_7d_shm[r, ])
        effect_size2[r] = wilcoxon_effect_size(dist2atlas_7d_inj[r, ], dist2atlas_28d_inj[r, ])
        effect_size3[r] = wilcoxon_effect_size(dist2atlas_28d_inj[r, ], dist2atlas_7d_inj[r, ])
        np_power[r] = nonparametric_power_ind(dist2atlas_7d_inj[r, ], dist2atlas_7d_shm[r, ])

    # Please Note that colorbars should go from 0 to 2 in your figure, 
    # but due to limitation of the nilearn functions, the data is scaled by a factor of 2
    plot_atlas_pval(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                    (1-np_power), out_fname=f'{dstdir}/rois_affected_np_power_nonparametric', alpha=1)
    plot_atlas_pval(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                    (2-np.abs(effect_size1))/2, out_fname=f'{dstdir}/rois_affected_effect_size_nonparametric', alpha=1)
    plot_atlas_pval(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                    (2-effect_size2)/2, out_fname=f'{dstdir}/rois_get_better_effect_size_nonparametric', alpha=1)
    plot_atlas_pval(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                    (2-effect_size3)/2, out_fname=f'{dstdir}/rois_get_worse_effect_size_nonparametric', alpha=1)

    # input('press any key')
