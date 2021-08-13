from tqdm import tqdm
import nilearn.image as ni
import numpy as np
from nilearn import plotting
from sklearn.cluster import KMeans


if __name__ == "__main__":

    n_subclusters = 2
    atlas_fname = '/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/01_study_specific_atlas_relabel.nii.gz'

    v = ni.load_img(atlas_fname)
    v_img = v.get_fdata()

    roi_list = np.unique(v_img)
    roi_list=np.setdiff1d(roi_list,[0])
    print(roi_list)

    for roi_id in tqdm(roi_list):
        roi_ind = v_img == roi_id
        xyz = np.where(roi_ind)
        xyz = np.vstack(xyz).T
        labels = KMeans(n_clusters=n_subclusters, random_state=0).fit_predict(xyz)
        v_img[roi_ind] = n_subclusters*v_img[roi_ind] + labels
 
    v = ni.new_img_like(v,np.int16(v_img))
    v.to_filename('/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats/01_study_specific_atlas_relabel_k' + str(n_subclusters) + '.nii.gz')
