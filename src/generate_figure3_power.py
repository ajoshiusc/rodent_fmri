#!/usr/bin/env python3

import os
import argparse
import numpy as np
import nilearn.image as ni
from nilearn import plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 14})

def main():
    parser = argparse.ArgumentParser(description='Generate Figure 3: Power Maps')
    parser.add_argument('--vdir', default='ratTBI/v4', help='Version output directory (e.g., ratTBI/v4)')
    parser.add_argument('--cmap', default='plasma', help='Colormap to use')
    parser.add_argument('--bg', default='/deneb_disk/ucla_mouse_injury/ucla_injury_rats/brain.nii.gz', help='Background MRI template')
    parser.add_argument('--out', default='figure3_power_maps.png', help='Output figure filename')
    
    args = parser.parse_args()

    # Base path assuming standard run_nonparametric.sh generated files
    base_dir = os.path.realpath(args.vdir)
    bg_img = args.bg

    # Methods mapped to their respective saved NIfTI files
    # The nonparametric codes save (1 - np_power) in the nifti files.
    methods = [
        ('BrainSync', f'{base_dir}/bsync_np/{args.cmap}/rois_affected_np_power_nonparametric_brainsync.nii.gz'),
        ('Connectivity\nProfile', f'{base_dir}/conn_np/{args.cmap}/rois_affected_np_power_nonparametric_conn.nii.gz'),
        ('Node\nDegree', f'{base_dir}/node_np/{args.cmap}/rois_affected_np_power_nonparametric_deg.nii.gz')
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'wspace': 0.1})

    for i, (title, path) in enumerate(methods):
        ax = axes[i]
        
        if not os.path.exists(path):
            print(f"Warning: File not found {path}. Run full nonparametric pipeline first.")
            ax.set_title(f"{title}\n[Data Missing]")
            ax.axis('off')
            continue

        # Load the data, which was stored as 1 - np_power
        img = ni.load_img(path)
        data = img.get_fdata()
        
        # Convert back to power directly
        # Regions outside the atlas were initialized to 1, so 1 - 1 = 0
        power_data = 1.0 - data
        power_img = ni.new_img_like(img, power_data)

        plotting.plot_stat_map(
            power_img,
            bg_img=bg_img,
            axes=ax,
            draw_cross=False,
            annotate=False,
            display_mode="y",
            cut_coords=[(111 - 90) * 1.25],
            cmap=args.cmap,
            colorbar=True,
            title=title,
            vmax=1.0,
            vmin=0.0
        )
        
        # Make the title aesthetic for a paper
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    plt.suptitle("Statistical Power (Mann-Whitney U)", fontsize=18, fontweight='bold', y=1.05)
    plt.savefig(args.out, dpi=600, bbox_inches='tight')
    print(f"Figure saved to {args.out}")
    plt.close()

if __name__ == "__main__":
    main()
