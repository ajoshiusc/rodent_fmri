from scipy import io as spio
import sys
sys.path.append('/ImagePTE1/ajoshi/code_farm/bfp/src/stats')
from brainsync import normalizeData
import os
import numpy as np
import nilearn.image as ni
from glob import glob

#from surfproc import patch_color_attrib, smooth_surf_function

#from dfsio import readdfs, writedfs


def get_roiwise_fmri(fmri, labels, label_ids):

    num_time = fmri.shape[3]
    num_rois = len(label_ids)
    rtseries = np.zeros((num_time, num_rois))

    labels = ni.load_img(labels).get_fdata()
    fmri = ni.load_img(fmri)


    for i, id in enumerate(label_ids):
        rtseries[:, i] = np.mean(fmri[labels == id,:], axis=0)

    rtseries_norm, _, _ = normalizeData(rtseries)

    return rtseries_norm, rtseries


if __name__ == "__main__":

    p_dir = '/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/inj_07d/'
    flist = glob(p_dir+'/at*.nii.gz')
    label_ids = np.arange(83,dtype=np.int16)
    # remove WM label from connectivity analysis
    label_ids = np.setdiff1d(label_ids, 0)

## Get a list of subjects
    sublist = list()    
    for f in flist:
        pth, fname = os.path.split(f)
        sublist.append(fname[6:26])

## Add extension for 7d
    sub_7d = sublist[1] + '_7d'
    sub_28d = sublist[1] + '_28d'

    fmri_7d = os.path.join(p_dir, sub_7d + '_rsfmri.nii.gz')
    atlas_labels_7d = os.path.join(p_dir, 'atlas_' + sub_7d + '_rsfmri.nii.gz')

    fmri_28d = os.path.join(p_dir, sub_28d + '_rsfmri.nii.gz')
    atlas_labels_28d = os.path.join(p_dir, 'atlas_' + sub_28d + '_rsfmri.nii.gz')


    fmri_roiwise_norm, fmri_roiwise = get_roiwise_fmri(fmri_data, anat_labels, label_ids)

    input('press any key')
