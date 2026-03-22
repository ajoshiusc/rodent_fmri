#!/usr/bin/env python3

import os
import argparse
import numpy as np
import nilearn.image as ni
from nilearn import plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 14})

def main():
    parser = argparse.ArgumentParser(description='Generate Figure 5: BrainSync detailed evaluation')
    parser.add_argument('--vdir', default='ratTBI/v4', help='Version output directory (e.g., ratTBI/v4)')
    parser.add_argument('--cmap', default='plasma', help='Colormap to use')
    parser.add_argument('--bg', default='/deneb_disk/ucla_mouse_injury/ucla_injury_rats/brain.nii.gz', help='Background MRI template')
    parser.add_argument('--out', default='figure5_brainsync_details.png', help='Output figure filename')
    
    args = parser.parse_args()

    # Base path
    base_dir = os.path.realpath(args.vdir)
    bg_img = args.bg
    alpha = 0.15  # Using a more lenient threshold for uncorrected results to show more patterns, as these are exploratory

    # Maps for BrainSync specifically (Uncorrected results)
    # A) rois_affected (7-day inj vs 7-day sham) -> Cross-sectional
    # B) rois_get_better (7-day inj vs 28-day inj: greater) -> Longitudinal recovery
    # C) rois_get_worse (7-day inj vs 28-day inj: less) -> Longitudinal decline
    
    methods = [
        ('(A) Affected ROIs\n(7d TBI vs Sham)', f'{base_dir}/bsync_np/{args.cmap}/rois_affected_nonparametric_uncorrected_brainsync.nii.gz'),
        ('(B) Functional Recovery\n(TBI: 7d to 28d)', f'{base_dir}/bsync_np/{args.cmap}/rois_get_better_nonparametric_uncorrected_brainsync.nii.gz'),
        ('(C) Functional Decline\n(TBI: 7d to 28d)', f'{base_dir}/bsync_np/{args.cmap}/rois_get_worse_nonparametric_uncorrected_brainsync.nii.gz')
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
        
        # Original visualization clips to alpha and subtracts from alpha
        # so high intensity = highly significant (p near 0)
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
        
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    plt.suptitle("Detailed Evaluation of Functional Alterations using BrainSync (Uncorrected $p \\leq 0.15$)", fontsize=18, fontweight='bold', y=1.05)
    
    plt.savefig(args.out, dpi=600, bbox_inches='tight')
    print(f"Figure saved to {args.out}")
    plt.close()

if __name__ == "__main__":
    main()
