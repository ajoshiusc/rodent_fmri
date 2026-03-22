#!/usr/bin/env python3

import os
import argparse
import nilearn.image as ni
from nilearn import plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 14})

def main():
    parser = argparse.ArgumentParser(description='Generate Figure S1: Connectivity Profile Variances / Distances to group atlas')
    parser.add_argument('--vdir', default='ratTBI/v4', help='Version output directory (e.g., ratTBI/v4)')
    parser.add_argument('--cmap', default='plasma', help='Colormap to use')
    parser.add_argument('--bg', default='/deneb_disk/ucla_mouse_injury/ucla_injury_rats/brain.nii.gz', help='Background MRI template')
    parser.add_argument('--out', default='figureS1_conn_distances.png', help='Output figure filename')
    
    args = parser.parse_args()

    # Base path
    base_dir = os.path.realpath(args.vdir)
    bg_img = args.bg
    
    # We want to plot the distance to group atlas (or variance) for:
    # 7-day Sham, 28-day Sham, 7-day TBI, 28-day TBI
    
    methods = [
        ('7-day Sham\n(Dist to Atlas)', f'{base_dir}/conn_np/{args.cmap}/var_7d_shm_nonparametric_conn.nii.gz'),
        ('28-day Sham\n(Dist to Atlas)', f'{base_dir}/conn_np/{args.cmap}/var_28d_shm_nonparametric_conn.nii.gz'),
        ('7-day TBI\n(Dist to Atlas)', f'{base_dir}/conn_np/{args.cmap}/var_7d_inj_nonparametric_conn.nii.gz'),
        ('28-day TBI\n(Dist to Atlas)', f'{base_dir}/conn_np/{args.cmap}/var_28d_inj_nonparametric_conn.nii.gz')
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'wspace': 0.1})

    for i, (title, path) in enumerate(methods):
        ax = axes[i]
        
        if not os.path.exists(path):
            print(f"Warning: File not found {path}. Run full nonparametric pipeline first.")
            ax.set_title(f"{title}\n[Data Missing]")
            ax.axis('off')
            continue

        # Load the Variance NIfTI
        img = ni.load_img(path)

        plotting.plot_stat_map(
            img,
            bg_img=bg_img,
            axes=ax,
            draw_cross=False,
            annotate=False,
            display_mode="y",
            cut_coords=[(111 - 90) * 1.25],
            cmap=args.cmap,
            colorbar=True,
            title=title,
            vmax=0.05, # Extracted from the default scaling param of the main script
            vmin=0.0
        )
        
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    plt.suptitle("Connectivity Profile: Average Functional Distances to Group Atlas", fontsize=18, fontweight='bold', y=1.05)
    
    plt.savefig(args.out, dpi=600, bbox_inches='tight')
    print(f"Figure saved to {args.out}")
    plt.close()

if __name__ == "__main__":
    main()
