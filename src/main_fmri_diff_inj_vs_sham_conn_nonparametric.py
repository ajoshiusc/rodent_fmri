#!/usr/bin/env python3

from logging import error
from scipy import io as spio
from scipy.stats import mannwhitneyu, wilcoxon
from scipy.stats import rankdata
from statsmodels.stats.multitest import fdrcorrection
import os
import csv
import argparse
from brainsync import normalizeData
import matplotlib.pyplot as plt
from nilearn import plotting
import numpy as np
import nilearn.image as ni
from glob import glob
from tqdm import tqdm

# from surfproc import patch_color_attrib, smooth_surf_function

# from dfsio import readdfs, writedfs

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
            _, p = mannwhitneyu(x_sim, y_sim, alternative='two-sided')
            if p < alpha:
                sig_count += 1
        except ValueError:
            pass
    return sig_count / n_sim

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
    flist = glob(dir_7d + "/at*.nii.gz")
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

    fmri_roiwise_7d_all = np.zeros([num_rois, num_rois, num_sub])
    fmri_roiwise_28d_all = np.zeros([num_rois, num_rois, num_sub])
    fmri_tdiff_all = np.zeros([num_rois, num_sub])

    for i, sub in enumerate(tqdm(sublist)):

        # Add extension for 7d
        sub_7d = sub

        sub_28d = sub_7d.replace("7d_rsfmri", "28d_rsfmri")
        sub_28d = sub_28d.replace("std_07", "std_28")

        f = glob(dir_28d + "/" + sub[:2] + "*.nii.gz")
        d, s = os.path.split(f[0])
        if len(f) != 1:
            error("error in 28th day timepoint files for " + s)

        sub_28d = s[:-7]

        fmri_7d = os.path.join(dir_7d, sub_7d + ".nii.gz")
        labels_7d = os.path.join(dir_7d, "atlas_" + sub_7d + ".nii.gz")

        fmri_28d = os.path.join(dir_28d, sub_28d + ".nii.gz")
        labels_28d = os.path.join(dir_28d, "atlas_" + sub_28d + ".nii.gz")

        data, _ = get_roiwise_fmri(fmri_7d, labels_7d, label_ids)
        fmri_roiwise_7d_all[:, :, i] = np.matmul((data.T), data)

        data, _ = get_roiwise_fmri(fmri_28d, labels_28d, label_ids)
        fmri_roiwise_28d_all[:, :, i] = np.matmul((data.T), data)

        fmri_tdiff_all[:, i] = np.linalg.norm(
            fmri_roiwise_7d_all[:, :, i] - fmri_roiwise_28d_all[:, :, i], axis=0
        )

    return fmri_tdiff_all, fmri_roiwise_7d_all, fmri_roiwise_28d_all

def plot_atlas_pval(atlas_image, atlas_labels, roi_ids, pval, out_fname, alpha=0.05,
                    cmap='hot',annotate=False,colorbar=False):

    atlas = ni.load_img(atlas_labels)
    atlas_img = atlas.get_fdata()

    img = np.ones(atlas.shape)

    for i, roi in enumerate(roi_ids):
        img[atlas_img == roi] = pval[i]

    pval_vol = ni.new_img_like(atlas, img)

    pval_vol.to_filename(out_fname + "_conn.nii.gz")

    img[img > alpha] = alpha
    pval_vol = ni.new_img_like(atlas, alpha - img)

    # plotting.plot_stat_map(bg_img=atlas_image, stat_map_img=pval_vol, vmax=alpha, threshold=0.0, output_file=out_fname + '_w.png',
    #                       draw_cross=False, annotate=True, display_mode="ortho", cut_coords=[(85-68)*1.25, (111-90)*1.25, (54-51)*1.25])
    plotting.plot_stat_map(
        bg_img=atlas_image,
        stat_map_img=pval_vol,
        vmax=alpha,
        threshold=0.0,
        output_file=out_fname + "_w_conn.png",
        draw_cross=False,
        annotate=annotate,
        display_mode="y",
        cut_coords=[(111 - 90) * 1.25],
        cmap=cmap,
        colorbar=colorbar,
        #vmin=0,
    )

    plt.show()

def plot_atlas_var(atlas_image, atlas_labels, roi_ids, roi_var, out_fname,
                    cmap='hot',annotate=False,colorbar=False, vmax=0.05):
    """Plot variance computed for each roi"""

    atlas = ni.load_img(atlas_labels)
    atlas_img = atlas.get_fdata()

    img = np.zeros(atlas.shape)

    for i, roi in enumerate(roi_ids):
        img[atlas_img == roi] = roi_var[i]

    val_vol = ni.new_img_like(atlas, img)

    val_vol.to_filename(out_fname + "_conn.nii.gz")
    val_vol = ni.new_img_like(atlas, img)

    # plot var

    # plotting.plot_stat_map(bg_img=atlas_image, stat_map_img=val_vol, threshold=0.0, output_file=out_fname + '_w.png', draw_cross=False,
    #                       annotate=True, display_mode="ortho", cut_coords=[(85-68)*1.25, (111-90)*1.25, (54-51)*1.25], vmax=0.001)
    plotting.plot_stat_map(
        bg_img=atlas_image,
        stat_map_img=val_vol,
        threshold=0.0,
        output_file=out_fname + "_w_conn.png",
        draw_cross=False,
        annotate=annotate,
        display_mode="y",
        cut_coords=[(111 - 90) * 1.25],
        vmax=vmax,
        vmin=0,
        cmap=cmap,
        colorbar=colorbar
        )

    plt.show()

if __name__ == "__main__":
    dstdir='/home/ajoshi/Desktop/rod_tbi/nonparametric_conn_results'
    srcdir='/deneb_disk/ucla_mouse_injury'
    parser = argparse.ArgumentParser(description='comparison of subjects in rodent fMRI study using nonparametric tests')
    parser.add_argument('--srcdir','-s', default=srcdir, help='source directory for data')
    parser.add_argument('--dstdir','-d', default=dstdir, help='output directory')
    parser.add_argument('--colorbar','-cb', action="store_true", help="include colorbar")
    parser.add_argument('--annotate','-a', action="store_true", help="annotate plots")
    parser.add_argument('--cmap','-c', default='hot', help='colormap')
    parser.add_argument('--vmax','-m', default=0.05, help='max range for variance maps', type=float)

    args = parser.parse_args()
    dstdir=os.path.realpath(args.dstdir)
    srcdir=os.path.realpath(args.srcdir)
    os.makedirs(dstdir, exist_ok=True)
    dir_7d = f'{srcdir}/ucla_injury_rats/shm_07d/'
    dir_28d = f'{srcdir}/ucla_injury_rats/shm_28d/'
    atlas_labels = f'{srcdir}/ucla_injury_rats/01_study_specific_atlas_relabel.nii.gz'
    atlas_image = f'{srcdir}/ucla_injury_rats/brain.nii.gz'

    ##
    fmri_tdiff_shm_all, fmri_shm_7d_all, fmri_shm_28d_all = get_fmri_diff_tpts(
        dir_7d, dir_28d
    )
    np.savez(f"{dstdir}/shm_nonparametric.npz", fmri_tdiff_shm_all=fmri_tdiff_shm_all)

    dir_7d = f'{srcdir}/ucla_injury_rats/inj_07d/'
    dir_28d = f'{srcdir}/ucla_injury_rats/inj_28d/'

    fmri_tdiff_inj_all, fmri_inj_7d_all, fmri_inj_28d_all = get_fmri_diff_tpts(
        dir_7d, dir_28d
    )
    np.savez(f"{dstdir}/inj_nonparametric.npz", fmri_tdiff_inj_all=fmri_tdiff_inj_all)

    num_rois = fmri_tdiff_inj_all.shape[0]
    pval2 = np.zeros(num_rois)
    pval = np.zeros(num_rois)
    pval_opp = np.zeros(num_rois)

    for r in tqdm(range(num_rois)):
        _, pval2[r] = mannwhitneyu(
            fmri_tdiff_inj_all[r,],
            fmri_tdiff_shm_all[r,],
            alternative="two-sided",
        )
        _, pval[r] = mannwhitneyu(
            fmri_tdiff_inj_all[r,],
            fmri_tdiff_shm_all[r,],
            alternative="less",
        )
        _, pval_opp[r] = mannwhitneyu(
            fmri_tdiff_inj_all[r,],
            fmri_tdiff_shm_all[r,],
            alternative="greater",
        )

    pval_orig = np.copy(pval)
    _, pval = fdrcorrection(pval, alpha=0.05)
    pval2_orig = np.copy(pval2)
    _, pval2 = fdrcorrection(pval2, alpha=0.05)
    pval_opp_orig = np.copy(pval_opp)
    _, pval_opp = fdrcorrection(pval_opp, alpha=0.05)

    np.savez(f"{dstdir}/pval_nonparametric.npz", pval2=pval2, pval=pval, pval_opp=pval_opp)
    print(np.stack((pval, pval2, pval_opp)).T)
    ##
    ##
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval,
        out_fname=f"{dstdir}/pval_7d_28d_nonparametric",
        alpha=0.25,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval_orig,
        out_fname=f"{dstdir}/pval_7d_28d_nonparametric_uncorrected",
        alpha=0.25,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval_orig,
        out_fname=f"{dstdir}/pval_7d_28d_nonparametric_uncorrected_alpha015",
        alpha=0.15,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval,
        out_fname=f"{dstdir}/pval_7d_28d_nonparametric_unthresholded",
        alpha=1.0,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval_orig,
        out_fname=f"{dstdir}/pval_7d_28d_nonparametric_uncorrected_unthresholded",
        alpha=1.0,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval2,
        out_fname=f"{dstdir}/pval2_7d_28d_nonparametric",
        alpha=0.25,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval2_orig,
        out_fname=f"{dstdir}/pval2_7d_28d_nonparametric_uncorrected",
        alpha=0.25,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval2_orig,
        out_fname=f"{dstdir}/pval2_7d_28d_nonparametric_uncorrected_alpha015",
        alpha=0.15,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval2,
        out_fname=f"{dstdir}/pval2_7d_28d_nonparametric_unthresholded",
        alpha=1.0,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval2_orig,
        out_fname=f"{dstdir}/pval2_7d_28d_nonparametric_uncorrected_unthresholded",
        alpha=1.0,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval_opp,
        out_fname=f"{dstdir}/pval_opp_7d_28d_nonparametric",
        alpha=0.25,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval_opp_orig,
        out_fname=f"{dstdir}/pval_opp_7d_28d_nonparametric_uncorrected",
        alpha=0.25,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval_opp_orig,
        out_fname=f"{dstdir}/pval_opp_7d_28d_nonparametric_uncorrected_alpha015",
        alpha=0.15,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval_opp,
        out_fname=f"{dstdir}/pval_opp_7d_28d_nonparametric_unthresholded",
        alpha=1.0,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval_opp_orig,
        out_fname=f"{dstdir}/pval_opp_7d_28d_nonparametric_uncorrected_unthresholded",
        alpha=1.0,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )

    ##
    # Calculate variance of 7d sham
    fmri_atlas_7d_shm = np.mean(fmri_shm_7d_all, axis=2)
    var_7d_shm = np.mean(
        (fmri_shm_7d_all - fmri_atlas_7d_shm[:, :, np.newaxis]) ** 2, axis=(0, 2)
    )
    
    plot_atlas_var(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        var_7d_shm,
        out_fname=f"{dstdir}/var_7d_shm_nonparametric",
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar,
        vmax=args.vmax
    )
    dist2atlas_7d_shm = np.sum(
        (fmri_shm_7d_all - fmri_atlas_7d_shm[:, :, np.newaxis]) ** 2, axis=(0)
    )

    ##
    # Calculate variance of 28d sham
    fmri_atlas = np.mean(fmri_shm_28d_all, axis=2)
    var_28d_shm = np.mean(
        (fmri_shm_28d_all - fmri_atlas[:, :, np.newaxis]) ** 2, axis=(0, 2)
    )

    plot_atlas_var(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        var_28d_shm,
        out_fname=f"{dstdir}/var_28d_shm_nonparametric",
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar,
        vmax=args.vmax
    )

    ##
    # Calculate variance of 7d inj
    fmri_atlas = np.mean(fmri_inj_7d_all, axis=2)
    var_7d_inj = np.mean(
        (fmri_inj_7d_all - fmri_atlas[:, :, np.newaxis]) ** 2, axis=(0, 2)
    )

    plot_atlas_var(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        var_7d_inj,
        out_fname=f"{dstdir}/var_7d_inj_nonparametric",
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar,
        vmax=args.vmax
    )

    # Calculate variance of 28d inj
    fmri_atlas = np.mean(fmri_inj_28d_all, axis=2)
    var_28d_inj = np.mean(
        (fmri_inj_28d_all - fmri_atlas[:, :, np.newaxis]) ** 2, axis=(0, 2)
    )

    plot_atlas_var(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        var_28d_inj,
        out_fname=f"{dstdir}/var_28d_inj_nonparametric",
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar,
        vmax=args.vmax
    )

    # Calculate variance of 28d shm wrt 7d shm grp atlas
    dist2atlas_28d_shm = np.sum(
        (fmri_shm_28d_all - fmri_atlas_7d_shm[:, :, np.newaxis]) ** 2, axis=(0)
    )
    var_28d_shm = np.mean(
        (fmri_shm_28d_all - fmri_atlas_7d_shm[:, :, np.newaxis]) ** 2,
        axis=(0, 2),
    )
    plot_atlas_var(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        var_28d_shm,
        out_fname=f"{dstdir}/var_28d_shm_7d_shm_nonparametric",
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar,
        vmax=args.vmax
    )

    # Calculate variance of 7d inj wrt 7d shm grp atlas
    dist2atlas_7d_inj = np.sum(
        (fmri_inj_7d_all - fmri_atlas_7d_shm[:, :, np.newaxis]) ** 2, axis=(0)
    )
    var_7d_inj = np.mean(
        (fmri_inj_7d_all - fmri_atlas_7d_shm[:, :, np.newaxis]) ** 2, axis=(0, 2)
    )
    plot_atlas_var(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        var_7d_inj,
        out_fname=f"{dstdir}/var_7d_inj_7d_shm_nonparametric",
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar,
        vmax=args.vmax
    )

    # Calculate variance of 28d inj wrt 7d shm grp atlas
    dist2atlas_28d_inj = np.sum(
        (fmri_inj_28d_all - fmri_atlas_7d_shm[:, :, np.newaxis]) ** 2, axis=(0)
    )
    var_28d_inj = np.mean(
        (fmri_inj_28d_all - fmri_atlas_7d_shm[:, :, np.newaxis]) ** 2,
        axis=(0, 2),
    )
    plot_atlas_var(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        var_28d_inj,
        out_fname=f"{dstdir}/var_28d_inj_7d_shm_nonparametric",
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar,
        vmax=args.vmax
    )

    ##
    # Which ROIs are affected in TBI: 7d inj vs 7d non-injury
    # Which ROIs get better: 7d inj vs 28d injury
    # Which ROIs get worse in TBI: 7d inj vs 28d injury

    pval = np.zeros(num_rois)
    pval2 = np.zeros(num_rois)
    pval3 = np.zeros(num_rois)

    for r in tqdm(range(num_rois)):
        _, pval[r] = mannwhitneyu(
            dist2atlas_7d_inj[r,],
            dist2atlas_7d_shm[r,],
            alternative="two-sided",
        )
        _, pval2[r] = wilcoxon(
            dist2atlas_7d_inj[r,], dist2atlas_28d_inj[r,], alternative="greater"
        )
        _, pval3[r] = wilcoxon(
            dist2atlas_7d_inj[r,], dist2atlas_28d_inj[r,], alternative="less"
        )

    pval_orig = np.copy(pval)
    _, pval = fdrcorrection(pval, alpha=0.05)
    pval2_orig = np.copy(pval2)
    _, pval2 = fdrcorrection(pval2, alpha=0.05)
    pval3_orig = np.copy(pval3)
    _, pval3 = fdrcorrection(pval3, alpha=0.05)

    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval,
        out_fname=f"{dstdir}/rois_affected_nonparametric",
        alpha=0.05,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval_orig,
        out_fname=f"{dstdir}/rois_affected_nonparametric_uncorrected",
        alpha=0.05,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval,
        out_fname=f"{dstdir}/rois_affected_nonparametric_unthresholded",
        alpha=1.0,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval_orig,
        out_fname=f"{dstdir}/rois_affected_nonparametric_uncorrected_unthresholded",
        alpha=1.0,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval2,
        out_fname=f"{dstdir}/rois_get_better_nonparametric",
        alpha=0.05,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval2_orig,
        out_fname=f"{dstdir}/rois_get_better_nonparametric_uncorrected",
        alpha=0.05,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval2,
        out_fname=f"{dstdir}/rois_get_better_nonparametric_unthresholded",
        alpha=1.0,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval2_orig,
        out_fname=f"{dstdir}/rois_get_better_nonparametric_uncorrected_unthresholded",
        alpha=1.0,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval3,
        out_fname=f"{dstdir}/rois_get_worse_nonparametric",
        alpha=0.05,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval3_orig,
        out_fname=f"{dstdir}/rois_get_worse_nonparametric_uncorrected",
        alpha=0.05,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval3,
        out_fname=f"{dstdir}/rois_get_worse_nonparametric_unthresholded",
        alpha=1.0,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        pval3_orig,
        out_fname=f"{dstdir}/rois_get_worse_nonparametric_uncorrected_unthresholded",
        alpha=1.0,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )

    ## Cohen's d
    effect_size1 = np.zeros(num_rois)
    effect_size2 = np.zeros(num_rois)
    effect_size3 = np.zeros(num_rois)
    np_power = np.zeros(num_rois)

    for r in tqdm(range(num_rois)):
        effect_size1[r] = cliffs_delta(dist2atlas_7d_inj[r,], dist2atlas_7d_shm[r,])
        effect_size2[r] = wilcoxon_effect_size(dist2atlas_7d_inj[r,], dist2atlas_28d_inj[r,])
        effect_size3[r] = wilcoxon_effect_size(dist2atlas_28d_inj[r,], dist2atlas_7d_inj[r,])
        np_power[r] = nonparametric_power_ind(dist2atlas_7d_inj[r,], dist2atlas_7d_shm[r,])

    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        (1 - np_power),
        out_fname=f"{dstdir}/rois_affected_np_power_nonparametric",
        alpha=1,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        (2 - np.abs(effect_size1)) / 2,
        out_fname=f"{dstdir}/rois_affected_effect_size_nonparametric",
        alpha=1,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        (2 - effect_size2) / 2,
        out_fname=f"{dstdir}/rois_get_better_effect_size_nonparametric",
        alpha=1,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )
    plot_atlas_pval(
        atlas_image,
        atlas_labels,
        np.arange(1, num_rois + 1),
        (2 - effect_size3) / 2,
        out_fname=f"{dstdir}/rois_get_worse_effect_size_nonparametric",
        alpha=1,
        cmap=args.cmap,annotate=args.annotate,colorbar=args.colorbar
    )

  #  input("press any key")
