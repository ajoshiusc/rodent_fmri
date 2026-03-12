#!/bin/bash

OD=${PWD}/ratTBI/v3/
install -d ${OD}

source .venv/bin/activate

DATADIR="/deneb_disk/ucla_mouse_injury"

for c in viridis plasma; do
    echo "============================================="
    echo "Running with colormap: $c"
    echo "============================================="

    # --- Conn ---
    time python3 src/main_fmri_diff_inj_vs_sham_conn_nonparametric.py -s $DATADIR -d ${OD}/conn_np/$c --colorbar -c $c -m 0.075
    time python3 src/main_fmri_diff_inj_vs_sham_conn.py               -s $DATADIR -d ${OD}/conn/$c --colorbar -c $c -m 0.075
    time python3 src/main_fmri_diff_inj_vs_sham_conn_log.py           -s $DATADIR -d ${OD}/conn_log/$c --colorbar -c $c -m 0.075

    # --- Node / Degree ---
    time python3 src/main_fmri_diff_inj_vs_sham_degree_nonparametric.py -s $DATADIR -d ${OD}/node_np/$c --colorbar -c $c -m 30
    time python3 src/main_fmri_diff_inj_vs_sham_degree.py               -s $DATADIR -d ${OD}/node/$c --colorbar -c $c -m 30
    time python3 src/main_fmri_diff_inj_vs_sham_degree_log.py           -s $DATADIR -d ${OD}/node_log/$c --colorbar -c $c -m 30

    # --- Bsync ---
    time python3 src/main_fmri_diff_inj_vs_sham_nonparametric.py -s $DATADIR -d ${OD}/bsync_np/$c --colorbar -c $c -m 0.001
    time python3 src/main_fmri_diff_inj_vs_sham.py               -s $DATADIR -d ${OD}/bsync/$c --colorbar -c $c -m 0.001
    time python3 src/main_fmri_diff_inj_vs_sham_log.py           -s $DATADIR -d ${OD}/bsync_log/$c --colorbar -c $c -m 0.001
done
