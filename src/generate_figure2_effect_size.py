#!/usr/bin/env python3

import os
import argparse
import numpy as np
import nilearn.image as ni
from nilearn import plotting
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 14})

def main():
    parser = argparse.ArgumentParser(description='Generate Figure 2: Effect Size Maps')
    parser.add_argument('--vdir', default='ratTBI/v4', help='Version output directory (e.g., ratTBI/v4)')
    parser.add_argument('--cmap', default='plasma', help='Colormap to use')
    parser.add_argument('--bg', default='/deneb_disk/ucla_mouse_injury/ucla_injury_rats/brain.nii.gz', help='Background MRI template')
    parser.add_argument('--out', default='figure2_effect_size_maps.png', help='Output figure filename')
    
    args = parser.parse_args()

    # Base path assuming standard run_nonparametric.sh generated files
    base_dir = os.path.realpath(args.vdir)
    bg_img = args.bg

    # Methods mapped to their respective saved NIfTI files
    # The output from script saved (2 - abs(effect_size)) / 2 into the NIfTI
    # Cliff's Delta goes from -1 to 1. Absolute effect size goes from 0 to 1.
    methods = [
        ('BrainSync', f'{base_dir}/bsync_np/{args.cmap}/rois_affected_effect_size_nonparametric_brainsync.nii.gz'),
        ('Connectivity\nProfile', f'{base_dir}/conn_np/{args.cmap}/rois_affected_effect_size_nonparametric_conn.nii.gz'),
        ('Node\nDegree', f'{base_dir}/node_np/{args.cmap}/rois_affected_effect_size_nonparametric_deg.nii.gz')
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'wspace': 0.1})

    for i, (title, path) in enumerate(methods):
        ax = axes[i]
        
        if not os.path.exists(path):
            print(f"Warning: File not found {path}. Run full nonparametric pipeline first.")
            ax.set_title(f"{title}\n[Data Missing]")
            ax.axis('off')
            continue

        # Load the data, which was stored as: (2 - abs(effect_size)) / 2
        img = ni.load_img(path)
        data = img.get_fdata()
        
        # Convert back to absolute effect size:
        # data = 1 - 0.5 * abs(es)  =>  0.5 * abs(es) = 1 - data  =>  abs(es) = 2 * (1 - data)
        es_data = 2.0 * (1.0 - data)
        es_img = ni.new_img_like(img, es_data)

        plotting.plot_stat_map(
            es_img,
            bg_img=bg_img,
            axes=ax,
            draw_cross=False,
            annotate=True,
            display_mode="y",
            cut_coords=[(111 - 90) * 1.25],
            cmap=args.cmap,
            vmax=1.0,
            colorbar=False,
            title=None,
            vmin=0.0
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)

    # Add a common colorbar to the right
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    plt.subplots_adjust(top=0.75, right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.6])
    cmap = plt.get_cmap(args.cmap)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)

    plt.suptitle("Nonparametric Effect Sizes (Cliff's Delta $|\\delta|$): 7-day TBI vs Sham", fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(args.out, dpi=600, bbox_inches='tight', facecolor='black')
    print(f"Figure saved to {args.out}")
    plt.close()

if __name__ == "__main__":
    main()
