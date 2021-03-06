from logging import error
from scipy import io as spio
from scipy.stats import ranksums, ttest_ind, ttest_rel
import os
import sys
sys.path.append('/ImagePTE1/ajoshi/code_farm/bfp/src/BrainSync')

from brainsync import normalizeData, brainSync, groupBrainSync
import matplotlib.pyplot as plt
from nilearn import plotting
import numpy as np
import nilearn.image as ni
from glob import glob
from tqdm import tqdm


#from surfproc import patch_color_attrib, smooth_surf_function

#from dfsio import readdfs, writedfs


def get_pointwise_fmri(fmri, atlas):

    labels = ni.load_img(atlas).get_fdata()
    fmri = ni.load_img(fmri).get_fdata()
    num_time = fmri.shape[3]
    num_vox = np.sum(labels.flatten()>0)
    rtseries = np.zeros((num_time, num_vox))

    rtseries = fmri[labels>0, :].T

    rtseries_norm, _, _ = normalizeData(rtseries)

    return rtseries_norm, rtseries, labels


def get_fmri_diff_tpts(dir_7d, dir_28d):
    flist = glob(dir_7d + '/at*.nii.gz')
    atlas = '/big_disk/ajoshi/ucla_mouse_injury/transforms_12dof_affine/atlas_labels_83_standard_space_6mm.nii.gz'

    num_time = 450

    # remove WM label from connectivity analysis
    num_vox = 1415 #11317 #label_ids.shape[0]
    num_sub = len(flist)


# Get a list of subjects
    sublist = list()
    for f in flist:
        pth, fname = os.path.split(f)
        sublist.append(fname[6:-7])  # 26

    fmri_pointwise_7d_all = np.zeros([num_time, num_vox, num_sub])
    fmri_pointwise_28d_all = np.zeros([num_time, num_vox, num_sub])
    fmri_tdiff_all = np.zeros([num_vox, num_sub])

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

        fmri_7d = os.path.join(dir_7d, 'warped6mm_' + sub_7d + '.nii.gz')

        fmri_28d = os.path.join(dir_28d,  'warped6mm_' + sub_28d + '.nii.gz')

        fmri_pointwise_7d_all[:, :, i], _, _ = get_pointwise_fmri(
            fmri_7d, atlas)
        fmri_pointwise_28d_all[:, :, i], _, _ = get_pointwise_fmri(
            fmri_28d, atlas)

        d, _ = brainSync(
            fmri_pointwise_7d_all[:, :, i], fmri_pointwise_28d_all[:, :, i])

        fmri_tdiff_all[:, i] = np.linalg.norm(
            fmri_pointwise_7d_all[:, :, i] - d, axis=0)

    return fmri_tdiff_all, fmri_pointwise_7d_all, fmri_pointwise_28d_all


def plot_atlas_pval_pointwise(atlas_fname, pval, out_fname, alpha=0.05):

    atlas = ni.load_img(atlas_fname)
    atlas_msk = atlas.get_data()>0
    img = np.ones(atlas.shape)
    img[atlas_msk] = pval 

    pval_vol = ni.new_img_like(atlas, img)

    pval_vol.to_filename(out_fname + '.nii.gz')

    img[img > alpha] = alpha
    pval_vol = ni.new_img_like(atlas, alpha - img)

    plotting.plot_stat_map(bg_img=atlas, stat_map_img=pval_vol, threshold=0.0, output_file=out_fname + '.png', draw_cross=False, annotate=True, display_mode="ortho", cut_coords=[(85-68)*1.25,(111-90)*1.25,(54-51)*1.25]) 
    plt.show()


def plot_atlas_var_pointwise(atlas_fname, fmri_var, out_fname):
    """ Plot variance computed for each roi """

    atlas = ni.load_img(atlas_fname)
    atlas_msk = atlas.get_data()>0
    img = np.zeros(atlas.shape)

    img[atlas_msk] = fmri_var

    val_vol = ni.new_img_like(atlas, img)

    val_vol.to_filename(out_fname + '.nii.gz')
    val_vol = ni.new_img_like(atlas, img)

    # plot var
    plotting.plot_stat_map(bg_img=atlas, stat_map_img=val_vol, threshold=0.0, output_file=out_fname + '.png', draw_cross=False, annotate=True, display_mode="ortho", cut_coords=[(85-68)*1.25,(111-90)*1.25,(54-51)*1.25], vmax=0.001)
    plt.show()

def fmri_sync(fmri,Os):
    """Sync gmri data using given Os""" 
    for j in range(fmri.shape[2]):
        fmri[:,:,j] = np.dot(Os[:, :, j], fmri[:, :, j])

    return fmri



if __name__ == "__main__":

    dir_7d = '/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/shm_07d/'
    dir_28d = '/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/shm_28d/'
    atlas_fname = '/big_disk/ajoshi/ucla_mouse_injury/transforms_12dof_affine/atlas_labels_83_standard_space_6mm.nii.gz'
##
    fmri_tdiff_shm_all, fmri_shm_7d_all, fmri_shm_28d_all = get_fmri_diff_tpts(
        dir_7d, dir_28d)
    np.savez('shm.npz', fmri_tdiff_inj_all=fmri_tdiff_shm_all)

    dir_7d = '/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/inj_07d/'
    dir_28d = '/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/inj_28d/'

    fmri_tdiff_inj_all, fmri_inj_7d_all, fmri_inj_28d_all = get_fmri_diff_tpts(
        dir_7d, dir_28d)
    np.savez('inj.npz', fmri_tdiff_inj_all=fmri_tdiff_inj_all)

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
    plot_atlas_pval_pointwise(atlas_fname, pval, out_fname='pval_7d_28d_pointwise_6mm', alpha=0.25)
    plot_atlas_pval_pointwise(atlas_fname, pval2, out_fname='pval2_7d_28d_pointwise_6mm', alpha=0.25)
    plot_atlas_pval_pointwise(atlas_fname, pval_opp, out_fname='pval_opp_7d_28d_pointwise_6mm', alpha=0.25)

##
    # Calculate variance of 7d sham
    a, Os, Costdif, TotalError = groupBrainSync(fmri_shm_7d_all)
    fmri_shm_7d_all_synced = fmri_sync(fmri_shm_7d_all, Os)
    fmri_atlas_7d_shm = np.mean(fmri_shm_7d_all_synced, axis=2)
    var_7d_shm = np.mean((fmri_shm_7d_all_synced - fmri_atlas_7d_shm[:, :,np.newaxis])**2, axis=(0, 2))
    plot_atlas_var_pointwise(atlas_fname, var_7d_shm, out_fname='var_7d_shm_pointwise_6mm')
    dist2atlas_7d_shm = np.sum((fmri_shm_7d_all_synced - fmri_atlas_7d_shm[:, :,np.newaxis])**2, axis=(0))
##
    # Calculate variance of 28d sham
    a, Os, Costdif, TotalError = groupBrainSync(fmri_shm_28d_all)
    fmri_shm_28d_all_synced = fmri_sync(fmri_shm_28d_all, Os)
    fmri_atlas = np.mean(fmri_shm_28d_all_synced, axis=2)
    var_28d_shm = np.mean((fmri_shm_28d_all_synced - fmri_atlas[:, :,np.newaxis])**2, axis=(0, 2))
    plot_atlas_var_pointwise(atlas_fname, var_28d_shm, out_fname='var_28d_shm_pointwise_6mm')

##
    # Calculate variance of 7d inj
    a, Os, Costdif, TotalError = groupBrainSync(fmri_inj_7d_all)
    fmri_inj_7d_all_synced = fmri_sync(fmri_inj_7d_all, Os)
    fmri_atlas = np.mean(fmri_inj_7d_all_synced, axis=2)
    var_7d_inj = np.mean((fmri_inj_7d_all_synced - fmri_atlas[:, :,np.newaxis])**2, axis=(0, 2))
    plot_atlas_var_pointwise(atlas_fname, var_7d_inj, out_fname='var_7d_inj_pointwise')

    # Calculate variance of 28d inj
    a, Os, Costdif, TotalError = groupBrainSync(fmri_inj_28d_all)
    fmri_inj_28d_all_synced = fmri_sync(fmri_inj_28d_all, Os)
    fmri_atlas = np.mean(fmri_inj_28d_all_synced, axis=2)
    var_28d_inj = np.mean((fmri_inj_28d_all_synced - fmri_atlas[:, :,np.newaxis])**2, axis=(0, 2))
    plot_atlas_var_pointwise(atlas_fname, var_28d_inj, out_fname='var_28d_inj_pointwise_6mm')

    # Calculate variance of 28d shm wrt 7d shm grp atlas
    num_sub = fmri_shm_28d_all.shape[2]
    fmri_shm_28d_all_synced = np.zeros(fmri_shm_28d_all.shape)

    for ind in range(num_sub):
        fmri_shm_28d_all_synced[:, :, ind], _ = brainSync(
            fmri_atlas_7d_shm, fmri_shm_28d_all[:, :, ind])

    dist2atlas_28d_shm = np.sum((fmri_shm_28d_all_synced - fmri_atlas_7d_shm[:, :,np.newaxis])**2, axis=(0))
    var_28d_shm = np.mean((fmri_shm_28d_all_synced - fmri_atlas_7d_shm[:, :,np.newaxis])**2, axis=(0, 2))
    plot_atlas_var_pointwise(atlas_fname, var_28d_shm, out_fname='var_28d_shm_7d_shm_pointwise_6mm')

    # Calculate variance of 7d inj wrt 7d shm grp atlas
    num_sub = fmri_inj_7d_all.shape[2]
    fmri_inj_7d_all_synced = np.zeros(fmri_inj_7d_all.shape)

    for ind in range(num_sub):
        fmri_inj_7d_all_synced[:, :, ind], _ = brainSync(
            fmri_atlas_7d_shm, fmri_inj_7d_all[:, :, ind])

    dist2atlas_7d_inj = np.sum((fmri_inj_7d_all_synced - fmri_atlas_7d_shm[:, :,np.newaxis])**2, axis=(0))
    var_7d_inj = np.mean((fmri_inj_7d_all_synced - fmri_atlas_7d_shm[:, :, np.newaxis])**2, axis=(0, 2))
    plot_atlas_var_pointwise(atlas_fname, var_7d_inj, out_fname='var_7d_inj_7d_shm_pointwise_6mm')

    # Calculate variance of 28d inj wrt 7d shm grp atlas
    num_sub = fmri_inj_28d_all.shape[2]
    fmri_inj_28d_all_synced = np.zeros(fmri_inj_28d_all.shape)

    for ind in range(num_sub):
        fmri_inj_28d_all_synced[:, :, ind], _ = brainSync(
            fmri_atlas_7d_shm, fmri_inj_28d_all[:, :, ind])
    
    dist2atlas_28d_inj = np.sum((fmri_inj_28d_all_synced - fmri_atlas_7d_shm[:, :,np.newaxis])**2, axis=(0))
    var_28d_inj = np.mean((fmri_inj_28d_all_synced - fmri_atlas_7d_shm[:, :,np.newaxis])**2, axis=(0, 2))
    plot_atlas_var_pointwise(atlas_fname, var_28d_inj, out_fname='var_28d_inj_7d_shm_pointwise_6mm')

    
## 
# Which ROIs are affected in TBI: 7d inj vs 7d non-injury
# Which ROIs get better: 7d inj vs 28d injury
# Which ROIs get worse in TBI: 7d inj vs 28d injury

    pval = np.zeros(num_rois)
    pval2 = np.zeros(num_rois)
    pval3 = np.zeros(num_rois)

    for r in tqdm(range(num_rois)):
        _, pval[r] = ttest_ind(dist2atlas_7d_inj[r, ], dist2atlas_7d_shm[r, ], alternative='greater', equal_var=False)
        _, pval2[r] = ttest_rel(dist2atlas_7d_inj[r, ], dist2atlas_28d_inj[r, ], alternative='greater') 
        _, pval3[r] = ttest_rel(dist2atlas_7d_inj[r, ], dist2atlas_28d_inj[r, ], alternative='less') 

    plot_atlas_pval_pointwise(atlas_fname, pval, out_fname='affected_pointwise_6mm', alpha=0.001)
    plot_atlas_pval_pointwise(atlas_fname, pval2, out_fname='get_better_pointwise_6mm', alpha=0.05)
    plot_atlas_pval_pointwise(atlas_fname, pval3, out_fname='get_worse_pointwise_6mm', alpha=0.05)

    
    input('press any key')
