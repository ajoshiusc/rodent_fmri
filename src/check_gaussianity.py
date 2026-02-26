import numpy as np
from scipy.stats import shapiro
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import os
import nilearn.image as ni
from nilearn import plotting
from main_fmri_diff_inj_vs_sham import plot_atlas_pval

def main():
    # Load distance measures
    # Assuming the script is run from the src directory or workspace root
    # We will use absolute paths or relative to workspace root
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    shm_file = os.path.join(workspace_root, 'shm.npz')
    
    if not os.path.exists(shm_file):
        print(f"Could not find {shm_file}")
        return
        
    shm_data = np.load(shm_file)['fmri_tdiff_shm_all']
    
    num_rois = shm_data.shape[0]
    
    shm_pvals = np.zeros(num_rois)
    shm_log_pvals = np.zeros(num_rois)
    
    alpha = 0.05
    
    for r in range(num_rois):
        # Check Gaussianity for shm using Shapiro-Wilk test
        stat, p_shm = shapiro(shm_data[r, :])
        shm_pvals[r] = p_shm
        
        # Check Gaussianity for log(shm)
        stat, p_shm_log = shapiro(np.log(shm_data[r, :]))
        shm_log_pvals[r] = p_shm_log
            
    # Apply FDR correction
    _, shm_pvals_fdr = fdrcorrection(shm_pvals, alpha=alpha)
    _, shm_log_pvals_fdr = fdrcorrection(shm_log_pvals, alpha=alpha)
            
    print(f"Number of non-Gaussian ROIs in shm group (FDR p < {alpha}): {np.sum(shm_pvals_fdr < alpha)}")
    print(f"Number of non-Gaussian ROIs in log(shm) group (FDR p < {alpha}): {np.sum(shm_log_pvals_fdr < alpha)}")
    
    srcdir = '/deneb_disk'
    atlas_labels = f'{srcdir}/ucla_mouse_injury/ucla_injury_rats/01_study_specific_atlas_relabel.nii.gz'
    atlas_image = f'{srcdir}/ucla_mouse_injury/ucla_injury_rats/brain.nii.gz'
    
    if not os.path.exists(atlas_labels) or not os.path.exists(atlas_image):
        print(f"Could not find atlas files in {srcdir}")
        return
        
    # Plot non-Gaussian ROIs on atlas
    # We plot the p-values directly, where p < 0.05 means non-Gaussian
    print("Plotting non-Gaussian ROIs for shm group...")
    plot_atlas_pval(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                    shm_pvals_fdr, out_fname=os.path.join(workspace_root, 'shm_gaussianity_pvals'), alpha=alpha)

    print("Plotting non-Gaussian ROIs for log(shm) group...")
    plot_atlas_pval(atlas_image, atlas_labels, np.arange(1, num_rois+1),
                    shm_log_pvals_fdr, out_fname=os.path.join(workspace_root, 'shm_log_gaussianity_pvals'), alpha=alpha)

if __name__ == "__main__":
    main()
