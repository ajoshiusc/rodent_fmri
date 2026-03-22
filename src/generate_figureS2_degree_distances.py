import os
import argparse
import matplotlib.pyplot as plt
from nilearn import plotting, image
import numpy as np

def generate_figureS2(vdir, bg):
    """
    Generate Figure S2: The average Node Degree functional distances between each of the
    sham/TBI groups at 7/28 days and the functional group atlas, computed for each ROI.
    """
    # Define the template for the background
    template_path = "ratTBI/v3/bsync/plasma/shm.npz"
    template_nii_path = template_path.replace('.npz', '_template.nii.gz')
    
    # We will use MNI152 template if the specific rat template isn't dynamically loaded, 
    # but we can just use the plotting default or a blank background if needed.
    # To be consistent with previous scripts, plot_stat_map works out of the box with the data.
    
    # Setup coordinates similar to other figures
    cut_coords = [(111 - 90) * 1.25]  # roughly y=26 in physical coordinates
    
    method_dir = os.path.join(vdir, "node_np", "plasma")
    
    # Files
    file_sham7 = os.path.join(method_dir, "var_7d_shm_nonparametric_deg.nii.gz")
    file_tbi7 = os.path.join(method_dir, "var_7d_inj_7d_shm_nonparametric_deg.nii.gz")
    file_sham28 = os.path.join(method_dir, "var_28d_shm_7d_shm_nonparametric_deg.nii.gz")
    file_tbi28 = os.path.join(method_dir, "var_28d_inj_7d_shm_nonparametric_deg.nii.gz")
    
    files = [
        (file_sham7, "Sham 7d"),
        (file_tbi7, "TBI 7d"),
        (file_sham28, "Sham 28d"),
        (file_tbi28, "TBI 28d"),
    ]
    
    # Calculate a global vmax across all images to ensure consistent color scale
    # We use 30 as it was provided to the original nonparametric execution scripts (-m 30)
    global_vmax = 30.0
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'wspace': 0.1})
    fig.suptitle("Figure S2: Average Node Degree Functional Distances to 7d Sham Atlas", fontsize=18, fontweight='bold', y=1.05)
    
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
        threshold = 1e-5

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

    plt.tight_layout()
    out_file = "figureS2_degree_distances.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure S2 for Node Degree Distances")
    parser.add_argument("--vdir", type=str, default="ratTBI/v4", help="Base directory for v4 outputs")
    parser.add_argument('--bg', default='/deneb_disk/ucla_mouse_injury/ucla_injury_rats/brain.nii.gz', help='Background MRI template')
    args = parser.parse_args()
    
    generate_figureS2(args.vdir, args.bg)
