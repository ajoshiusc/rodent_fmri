from tqdm import tqdm
from glob import glob
import nilearn.image as ni
import numpy as np
import os
import sys

sys.path.append('/ImagePTE1/ajoshi/code_farm/bfp/src/BrainSync')

from brainsync import normalizeData, brainSync
from logging import error
from scipy import io as spio

#from surfproc import patch_color_attrib, smooth_surf_function

#from dfsio import readdfs, writedfs


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


if __name__ == "__main__":

    dir_7d = '/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/inj_07d/'
    dir_28d = '/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/inj_28d/'

    flist = glob(dir_7d + '/at*.nii.gz')
    label_ids = np.arange(83, dtype=np.int16)
    # remove WM label from connectivity analysis
    label_ids = np.setdiff1d(label_ids, 0)
    num_time = 450
    num_rois = label_ids.shape[0]
    num_sub = len(flist)


# Get a list of subjects
    sublist = list()
    for f in flist:
        pth, fname = os.path.split(f)
        sublist.append(fname[6:26])

    fmri_roiwise_7d_all = np.zeros([num_time, num_rois, num_sub])
    fmri_roiwise_28d_all = np.zeros([num_time, num_rois, num_sub])
    fmri_tdiff_all = np.zeros([num_rois, num_sub])

    for i, sub in enumerate(tqdm(sublist)):

        # Add extension for 7d
        sub_7d = sub + '_7d'

        f = glob(dir_28d + '/' + sub[:2] + '*.nii.gz')
        d, s = os.path.split(f[0])
        if len(f) != 1:
            error('error in 28th day timepoint files for ' + s)

        sub_28d = s[:24]

        fmri_7d = os.path.join(dir_7d, sub_7d + '_rsfmri.nii.gz')
        labels_7d = os.path.join(dir_7d, 'atlas_' + sub_7d + '_rsfmri.nii.gz')

        fmri_28d = os.path.join(dir_28d, sub_28d + '_rsfmri.nii.gz')
        labels_28d = os.path.join(
            dir_28d, 'atlas_' + sub_28d + '_rsfmri.nii.gz')

        fmri_roiwise_7d_all[:, :, i], _ = get_roiwise_fmri(
            fmri_7d, labels_7d, label_ids)
        fmri_roiwise_28d_all[:, :, i], _ = get_roiwise_fmri(
            fmri_28d, labels_28d, label_ids)

        d, _ = brainSync(
            fmri_roiwise_7d_all[:, :, i], fmri_roiwise_28d_all[:, :, i])

        fmri_tdiff_all[:, i] = np.linalg.norm(
            fmri_roiwise_7d_all[:, :, i] - d, axis=0)

    np.savez('inj.npz', fmri_tdiff_all=fmri_tdiff_all)

    input('press any key')
