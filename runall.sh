#!/bin/bash

OD=${PWD}/ratTBI/v1/

install -d ${OD}
for c in viridis plasma; do
    time ./src/main_fmri_diff_inj_vs_sham_conn.py -s ~/Data/TBI/ucla_rodent_tbi_data_from_Neil/ucla_mouse_injury -d ${OD}/conn/$c --colorbar -c $c -m 0.075
    time ./src/main_fmri_diff_inj_vs_sham_degree.py -s ~/Data/TBI/ucla_rodent_tbi_data_from_Neil/ucla_mouse_injury -d ${OD}/node/$c --colorbar -c $c -m 30
    time ./src/main_fmri_diff_inj_vs_sham.py -s ~/Data/TBI/ucla_rodent_tbi_data_from_Neil/ucla_mouse_injury -d ${OD}/bsync/$c --colorbar -c $c -m 0.001
done
