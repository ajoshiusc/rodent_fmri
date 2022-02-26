# rodent_fmri

`main_apply_xfm.py` warps subject fmri data to atlas space for pointwise analysis

`main_fmri_diff_inj_vs_sham.py` roi-wise analysis of fmri data using brainsyc. This compares injury subjects and sham subject to an atlas created using sham subjects.

`main_fmri_diff_inj_vs_sham_conn.py` roi-wise analysis of fmri data using connectivity feature. This compares injury subjects and sham subject to an atlas created using sham subjects.

`main_fmri_diff_inj_vs_sham_degree.py` pointwise-wise analysis of fmri data using node degree. Calculates and plots variance for sham and inj datasets for each point in the brain.

`main_fmri_diff_inj_vs_sham_Ftest.py` F-test to test equality of variance for each roi for sham and injury cohorts. We expect that injury cohort have more heterogeneous functional connectivity, and therefore have higher variance.

