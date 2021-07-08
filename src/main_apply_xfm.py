import time
import os
from glob import glob
from tqdm import tqdm

atlas = '/big_disk/ajoshi/ucla_mouse_injury/transforms_12dof_affine/atlas_labels_83_standard_space.nii.gz'

subdir = 'shm_07d'

flist = glob(os.path.join('/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats', subdir) + '/at*.nii.gz')

num_sub = len(flist)

# Get a list of subjects
sublist = list()
for f in flist:
    pth, fname = os.path.split(f)
    sublist.append(fname[6:-7])  # 26


for subbase in tqdm(sublist):
    subfile = os.path.join('/big_disk/ajoshi/ucla_mouse_injury/ucla_injury_rats', subdir,  subbase + '.nii.gz')
    xfmfile = os.path.join('/big_disk/ajoshi/ucla_mouse_injury/transforms_12dof_affine', subdir, subbase + '.mat')

    pth, fname = os.path.split(subfile)
    outfile = os.path.join(pth,  'warped_' + fname)

    cmd = 'flirt -in ' + subfile + ' -ref ' + atlas + ' -applyxfm -init ' + xfmfile + ' -out ' + outfile

    t1 = time.time()
    os.system(cmd)

    t2 = time.time()

print('done')

