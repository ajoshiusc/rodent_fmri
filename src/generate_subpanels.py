#!/usr/bin/env python3

import os
import argparse
import numpy as np
import nilearn.image as ni
from nilearn import plotting
import matplotlib.pyplot as plt

def save_panel(img_path, out_path, bg_img, cmap='plasma', vmax=None, vmin=0.0, threshold=1e-6, transform=None, alpha=None):
    """
    Load a NIfTI image, apply optional transformations, and save a 'clean' 
    MRI slice as a PNG (no colorbar, no text, no annotations).
    """
    if not os.path.exists(img_path):
        print(f"Warning: File not found {img_path}. Skipping {out_path}.")
        return

    img = ni.load_img(img_path)
    data = img.get_fdata()

    # Apply specific transformations used in original figure scripts
    if transform == 'effect_size':
        # data = 1 - 0.5 * abs(es)  =>  abs(es) = 2 * (1 - data)
        data = 2.0 * (1.0 - data)
    elif transform == 'power':
        # data = 1 - power => power = 1 - data
        data = 1.0 - data
    elif transform == 'p_value' and alpha is not None:
        # p-values: high significance (low p) should be bright.
        data[data > alpha] = alpha
        data = alpha - data
    
    plot_img = ni.new_img_like(img, data)

    # Create a figure for this specific panel
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    plotting.plot_stat_map(
        plot_img,
        bg_img=bg_img,
        axes=ax,
        draw_cross=False,
        annotate=False,
        display_mode="y",
        cut_coords=[(111 - 90) * 1.25],
        cmap=cmap,
        colorbar=False,
        vmax=vmax,
        vmin=vmin,
        threshold=threshold,
        title=None
    )
    
    # Save with minimal padding and no text
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Successfully saved {out_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate clean individual subpanels for all figures.')
    parser.add_argument('--vdir', default='ratTBI/v4', help='Version output directory (e.g., ratTBI/v4)')
    parser.add_argument('--bg', default='/deneb_disk/ucla_mouse_injury/ucla_injury_rats/brain.nii.gz', help='Background MRI template')
    parser.add_argument('--cmap', default='plasma', help='Colormap to use')
    
    args = parser.parse_args()
    base_dir = os.path.realpath(args.vdir)
    bg_img = args.bg

    # --- Figure 1: BrainSync functional distances ---
    f1_dir = os.path.join(base_dir, "bsync_np", args.cmap)
    f1_panels = [
        (os.path.join(f1_dir, "var_7d_shm_nonparametric.nii.gz"), "Fig1A.png"),
        (os.path.join(f1_dir, "var_7d_inj_7d_shm_nonparametric.nii.gz"), "Fig1B.png"),
        (os.path.join(f1_dir, "var_28d_shm_7d_shm_nonparametric.nii.gz"), "Fig1C.png"),
        (os.path.join(f1_dir, "var_28d_inj_7d_shm_nonparametric.nii.gz"), "Fig1D.png"),
    ]
    for path, out in f1_panels:
        save_panel(path, out, bg_img, vmax=0.001, threshold=1e-6)

    # --- Figure 2: Effect Size Maps ---
    # Conversions needed for A:BrainSync, B:Conn, C:Node
    f2_configs = [
        (os.path.join(base_dir, "bsync_np", args.cmap, "rois_affected_effect_size_nonparametric_brainsync.nii.gz"), "Fig2A.png"),
        (os.path.join(base_dir, "conn_np", args.cmap, "rois_affected_effect_size_nonparametric_conn.nii.gz"), "Fig2B.png"),
        (os.path.join(base_dir, "node_np", args.cmap, "rois_affected_effect_size_nonparametric_deg.nii.gz"), "Fig2C.png"),
    ]
    for path, out in f2_configs:
        save_panel(path, out, bg_img, vmax=1.0, transform='effect_size', threshold=1e-6)

    # --- Figure 3: Power Maps ---
    f3_configs = [
        (os.path.join(base_dir, "bsync_np", args.cmap, "rois_affected_np_power_nonparametric_brainsync.nii.gz"), "Fig3A.png"),
        (os.path.join(base_dir, "conn_np", args.cmap, "rois_affected_np_power_nonparametric_conn.nii.gz"), "Fig3B.png"),
        (os.path.join(base_dir, "node_np", args.cmap, "rois_affected_np_power_nonparametric_deg.nii.gz"), "Fig3C.png"),
    ]
    for path, out in f3_configs:
        save_panel(path, out, bg_img, vmax=1.0, transform='power', threshold=1e-6)

    # --- Figure 4: ROI p-values ---
    f4_configs = [
        (os.path.join(base_dir, "bsync_np", args.cmap, "rois_affected_nonparametric_brainsync.nii.gz"), "Fig4A.png"),
        (os.path.join(base_dir, "conn_np", args.cmap, "rois_affected_nonparametric_conn.nii.gz"), "Fig4B.png"),
        (os.path.join(base_dir, "node_np", args.cmap, "rois_affected_nonparametric_deg.nii.gz"), "Fig4C.png"),
    ]
    for path, out in f4_configs:
        save_panel(path, out, bg_img, vmax=0.05, transform='p_value', alpha=0.05, threshold=1e-6)

    # --- Figure 5: BrainSync Details (Uncorrected) ---
    f5_configs = [
        (os.path.join(base_dir, "bsync_np", args.cmap, "rois_affected_nonparametric_uncorrected_brainsync.nii.gz"), "Fig5A.png"),
        (os.path.join(base_dir, "bsync_np", args.cmap, "rois_get_better_nonparametric_uncorrected_brainsync.nii.gz"), "Fig5B.png"),
        (os.path.join(base_dir, "bsync_np", args.cmap, "rois_get_worse_nonparametric_uncorrected_brainsync.nii.gz"), "Fig5C.png"),
    ]
    for path, out in f5_configs:
        save_panel(path, out, bg_img, vmax=0.15, transform='p_value', alpha=0.15, threshold=1e-6)

    # --- Supplemental Figure S1: Connectivity Distances ---
    fS1_dir = os.path.join(base_dir, "conn_np", args.cmap)
    fS1_panels = [
        (os.path.join(fS1_dir, "var_7d_shm_nonparametric_conn.nii.gz"), "FigS1A.png"),
        (os.path.join(fS1_dir, "var_28d_shm_nonparametric_conn.nii.gz"), "FigS1B.png"),
        (os.path.join(fS1_dir, "var_7d_inj_nonparametric_conn.nii.gz"), "FigS1C.png"),
        (os.path.join(fS1_dir, "var_28d_inj_nonparametric_conn.nii.gz"), "FigS1D.png"),
    ]
    for path, out in fS1_panels:
        save_panel(path, out, bg_img, vmax=0.05, threshold=1e-6)

    # --- Supplemental Figure S2: Node Degree Distances ---
    fS2_dir = os.path.join(base_dir, "node_np", args.cmap)
    fS2_panels = [
        (os.path.join(fS2_dir, "var_7d_shm_nonparametric_deg.nii.gz"), "FigS2A.png"),
        (os.path.join(fS2_dir, "var_7d_inj_7d_shm_nonparametric_deg.nii.gz"), "FigS2B.png"),
        (os.path.join(fS2_dir, "var_28d_shm_7d_shm_nonparametric_deg.nii.gz"), "FigS2C.png"),
        (os.path.join(fS2_dir, "var_28d_inj_7d_shm_nonparametric_deg.nii.gz"), "FigS2D.png"),
    ]
    for path, out in fS2_panels:
        save_panel(path, out, bg_img, vmax=30.0, threshold=1e-5)

    # --- Supplemental Figure S3: Conn Profile p-values ---
    fS3_dir = os.path.join(base_dir, "conn_np", args.cmap)
    fS3_panels = [
        (os.path.join(fS3_dir, "rois_affected_nonparametric_uncorrected_alpha015_conn.nii.gz"), "FigS3A.png"),
        (os.path.join(fS3_dir, "rois_get_better_nonparametric_uncorrected_alpha015_conn.nii.gz"), "FigS3B.png"),
        (os.path.join(fS3_dir, "rois_get_worse_nonparametric_uncorrected_alpha015_conn.nii.gz"), "FigS3C.png"),
    ]
    for path, out in fS3_panels:
        save_panel(path, out, bg_img, vmax=0.15, transform='p_value', alpha=0.15, threshold=1e-6)

    # --- Supplemental Figure S4: Node Degree p-values ---
    fS4_dir = os.path.join(base_dir, "node_np", args.cmap)
    fS4_panels = [
        (os.path.join(fS4_dir, "rois_affected_nonparametric_uncorrected_alpha015_deg.nii.gz"), "FigS4A.png"),
        (os.path.join(fS4_dir, "rois_get_better_nonparametric_uncorrected_alpha015_deg.nii.gz"), "FigS4B.png"),
        (os.path.join(fS4_dir, "rois_get_worse_nonparametric_uncorrected_alpha015_deg.nii.gz"), "FigS4C.png"),
    ]
    for path, out in fS4_panels:
        save_panel(path, out, bg_img, vmax=0.15, transform='p_value', alpha=0.15, threshold=1e-6)


    print("All requested subpanels generated.")

if __name__ == "__main__":
    main()
