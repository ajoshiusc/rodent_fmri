#!/usr/bin/env python3

import os
import argparse
import numpy as np
import nilearn.image as ni
from nilearn import plotting
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Generate Figure 4: ROI-wise p-value maps')
    parser.add_argument('--vdir', default='ratTBI/v4', help='Version output directory (e.g., ratTBI/v4)')
    parser.add_argument('--cmap', default='plasma', help='Colormap to use')
    parser.add_argument('--bg', default='/deneb_disk/ucla_mouse_injury/ucla_injury_rats/brain.nii.gz', help='Background MRI template')
    parser.add_argument('--out', default='figure4_pval_maps.png', help='Output figure filename')
    
    args = parser.parse_args()

    # Base path assuming standard run_nonparametric.sh generated files
    base_dir = os.path.realpath(args.vdir)
    bg_img = args.bg
    alpha = 0.05

    # Methods mapped to their respective saved NIfTI files
    # rois_affected_nonparametric represent the 7-day inj vs 7-day sham FDR-corrected p-values
    methods = [
        ('BrainSync', f'{base_dir}/bsync_np/{args.cmap}/rois_affected_nonparametric_brainsync.nii.gz'),
        ('Connectivity\nProfile', f'{base_dir}/conn_np/{args.cmap}/rois_affected_nonparametric_conn.nii.gz'),
        ('Node\nDegree', f'{base_dir}/node_np/{args.cmap}/rois_affected_nonparametric_deg.nii.gz')
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'wspace': 0.1})

    for i, (title, path) in enumerate(methods):
        ax = axes[i]
        
        if not os.path.exists(path):
            print(f"Warning: File not found {path}. Run full nonparametric pipeline first.")
            ax.set_title(f"{title}\n[Data Missing]")
            ax.axis('off')
            continue

        # Load the NIfTI which stores the actual FDR corrected p-values
        img = ni.load_img(path)
        data = img.get_fdata()
        
        # Just like original script visualization, we want to only show <= 0.05,
        # with the lowest p-values shining bright. So we limit to alpha and subtract from alpha.
        data[data > alpha] = alpha
        plot_data = alpha - data
        
        pval_img = ni.new_img_like(img, plot_data)

        plotting.plot_stat_map(
            pval_img,
            bg_img=bg_img,
            axes=ax,
            draw_cross=False,
            annotate=False,
            display_mode="y",
            cut_coords=[(111 - 90) * 1.25],
            cmap=args.cmap,
            colorbar=True,
            title=title,
            vmax=alpha,
            threshold=0.0
        )
        
        # Clean title over text formatting issues from overlapping axes
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    plt.suptitle("Functional alterations 7-day TBI vs Sham (FDR-corrected $p \\leq 0.05$)", fontsize=18, fontweight='bold', y=1.05)
    
    # Save formatting to avoid UserWarnings from nilearn and tight layout discrepancies
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {args.out}")
    plt.close()

if __name__ == "__main__":
    main()
