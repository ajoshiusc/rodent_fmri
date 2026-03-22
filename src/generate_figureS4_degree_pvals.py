import os
import argparse
import numpy as np
import nilearn.image as ni
from nilearn import plotting
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 14})

def main():
    parser = argparse.ArgumentParser(description='Generate Figure S4: Node Degree profile statistical testing')
    parser.add_argument('--vdir', default='ratTBI/v4', help='Version output directory (e.g., ratTBI/v4)')
    parser.add_argument('--cmap', default='plasma', help='Colormap to use')
    parser.add_argument('--bg', default='/deneb_disk/ucla_mouse_injury/ucla_injury_rats/brain.nii.gz', help='Background MRI template')
    parser.add_argument('--out', default='figureS4_degree_pvals.png', help='Output figure filename')
    args = parser.parse_args()

    # Base path
    base_dir = os.path.realpath(args.vdir)
    bg_img = args.bg
    alpha = 0.15
    
    # Files matching the prompt specification for node degree profile
    # For small sample size (N=7), we map the uncorrected array with an alpha of 0.15 
    methods = [
        ('(A) 7d TBI vs 7d Sham\n(Mann-Whitney U)', f'{base_dir}/node_np/{args.cmap}/rois_affected_nonparametric_uncorrected_alpha015_deg.nii.gz'),
        ('(B) 28d vs 7d Recovery\n(Wilcoxon signed-rank)', f'{base_dir}/node_np/{args.cmap}/rois_get_better_nonparametric_uncorrected_alpha015_deg.nii.gz'),
        ('(C) 28d vs 7d Decline\n(Wilcoxon signed-rank)', f'{base_dir}/node_np/{args.cmap}/rois_get_worse_nonparametric_uncorrected_alpha015_deg.nii.gz')
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'wspace': 0.1})

    for i, (title, path) in enumerate(methods):
        ax = axes[i]
        
        if not os.path.exists(path):
            print(f"Warning: File not found {path}. Run full nonparametric pipeline first.")
            ax.set_title(f"{title}\n[Data Missing]")
            ax.axis('off')
            continue

        # Load the NIfTI which stores the p-values
        img = ni.load_img(path)
        data = img.get_fdata()
        
        # We limit to alpha and subtract from alpha to make smaller (more significant) p-values brighter.
        data[data > alpha] = alpha
        plot_data = alpha - data
        
        # Prevent completely blank brains rendering black if there is no significant data
        if np.max(plot_data) == 0:
             print(f"Note: No values passed the correction for {title}. Background will be shown.")
        
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
            threshold=1e-6 # small threshold so background shows through zeros
        )
        
        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    plt.suptitle("Figure S4: Node Degree Statistical Testing (Uncorrected $p \\leq 0.15$)", fontsize=18, fontweight='bold', y=1.05)
    plt.savefig(args.out, dpi=600, bbox_inches='tight')
    print(f"Figure saved to {args.out}")
    plt.close()

if __name__ == "__main__":
    main()