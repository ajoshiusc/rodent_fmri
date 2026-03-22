import os
import argparse
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 14})
from nilearn import plotting, image
import numpy as np

def generate_figure1(vdir, bg):
    """
    Generate Figure 1: The average BrainSync functional distances between each of the
    sham/TBI groups at 7/28 days and the functional group atlas, computed for each ROI.
    """
    # Setup coordinates similar to other figures
    cut_coords = [(111 - 90) * 1.25]  # roughly y=26 in physical coordinates
    
    method_dir = os.path.join(vdir, "bsync_np", "plasma")
    
    # Files mapped to their respective groups synced to the 7-day Sham atlas
    file_sham7 = os.path.join(method_dir, "var_7d_shm_nonparametric.nii.gz")
    file_tbi7 = os.path.join(method_dir, "var_7d_inj_7d_shm_nonparametric.nii.gz")
    file_sham28 = os.path.join(method_dir, "var_28d_shm_7d_shm_nonparametric.nii.gz")
    file_tbi28 = os.path.join(method_dir, "var_28d_inj_7d_shm_nonparametric.nii.gz")
    
    files = [
        (file_sham7, "7-day Sham\n(Dist to Atlas)"),
        (file_tbi7, "7-day TBI\n(Dist to Atlas)"),
        (file_sham28, "28-day Sham\n(Dist to Atlas)"),
        (file_tbi28, "28-day TBI\n(Dist to Atlas)"),
    ]
    
    # We use 0.001 as it was provided to the original nonparametric execution scripts (-m 0.001)
    global_vmax = 0.001
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'wspace': 0.1})
    fig.suptitle("Figure 1: Average BrainSync Functional Distances to Functional Group Atlas", fontsize=18, fontweight='bold', y=1.05)
    
    for i, (fpath, title) in enumerate(files):
        ax = axes[i]
        
        if not os.path.exists(fpath):
            print(f"Warning: File not found: {fpath}")
            ax.set_title(f"{title}\n(File missing)")
            ax.axis('off')
            continue
            
        img = image.load_img(fpath)
        data = img.get_fdata()
        
        # we need a threshold so zero values become transparent and background shows
        threshold = 1e-6

        plotting.plot_stat_map(
            img, 
            display_mode='y', 
            cut_coords=cut_coords, 
            axes=ax, 
            cmap='plasma', 
            vmax=global_vmax,
            threshold=threshold,
            colorbar=True,
            annotate=False,
            draw_cross=False,
            bg_img=bg
        )
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    out_file = "figure1_brainsync_distances.png"
    plt.savefig(out_file, dpi=600, bbox_inches='tight')
    print(f"Figure saved to {out_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 1 for BrainSync Distances")
    parser.add_argument("--vdir", type=str, default="ratTBI/v4", help="Base directory for v4 outputs")
    parser.add_argument('--bg', default='/deneb_disk/ucla_mouse_injury/ucla_injury_rats/brain.nii.gz', help='Background MRI template')
    args = parser.parse_args()
    
    generate_figure1(args.vdir, args.bg)