import os
import glob
import zarr
import numpy as np
import scipy.io as sio
from scipy.io import savemat
from skimage.registration import phase_cross_correlation
from multiprocessing import Pool
from matplotlib import image


ZARR_FILENAME = "/storage1/fs1/jlmorgan/Active/morganLab/DATA/LGN_Developing/KxR_P11LGN/diced/KxR_P11LGN_mip0_s750_799.zarr"
VASTSUBS_FILENAME = "/storage1/fs1/jlmorgan/Active/morganLab/DATA/LGN_Developing/KxR_P11LGN/CellNav_KxR/vastSubsPlanes749-759.mat"
MAX_NEXT_PLANES = 10 # maximum number of adjacent planes containing mitochondria for an RGC bouton
PATCH_SIZE = 512
ADJACENT_STICHED_PATCHES_SIZE = (3 * PATCH_SIZE, 3 * PATCH_SIZE) # the size is equal to the size of three adjacent 512*512 patches
DS_FACTOR = 16 # 2 ** MIPlEVEL (4)

NEIGHBORHOOD = [
    [-1, -1], [-1, 0], [-1, 1], 
    [0, -1], [0, 0], [0, 1], 
    [1, -1], [1, 0], [1, 1]
]
SLICES = range(750, 761)

vastSubs = sio.loadmat(VASTSUBS_FILENAME)["vastSubs"][0]
vastSubs_range = range(len(vastSubs))
zarr_f = zarr.open(ZARR_FILENAME, 'r')


def collect_samples_for_slice(slice_num):
    samples = list(zarr_f[f"{slice_num}/raw"].keys())
    samples = filter(str.isdigit, samples)
    samples = sorted(map(int, samples))

    rc = [
        {
            "sample": sample,
            "rowCol": np.array(zarr_f[row_col_key][:] if row_col_key in zarr_f else sample.split("_")).astype(int) 
        }
        for sample in samples
        if (row_col_key := f"{slice_num}/xy_coordinates/{sample}")
    ]
                
    return {"slice": int(slice_num), "rowColList": rc}

# Function to process shift and save images for a given slice
def process_shift_and_save_image(slice_data):
    id, rc_list = slice_data
    rows, cols, zs = vastSubs[id][:, 0], vastSubs[id][:, 1], vastSubs[id][:, 2]
    min_row = min(rows)
    min_col = min(cols)
    slice_num = np.unique(zs)
    slice_num = slice_num[0] + 1
    print(slice_num, ".................................")
    
    shifts = []
    rc = next((item["rowColList"] for item in rc_list if item["slice"] == slice_num), [])
    row_col_array = np.array([entry["rowCol"] for entry in rc]) if rc else np.empty((0, 2))

    planes = []
    for s in range(slice_num, slice_num + MAX_NEXT_PLANES):

        patch = np.zeros(ADJACENT_STICHED_PATCHES_SIZE)
        if rc:
            min_row, min_col = rc[0]["rowCol"]
            patch_row = (min_row * DS_FACTOR) // PATCH_SIZE
            patch_col = (min_col * DS_FACTOR) // PATCH_SIZE

            for i, neighbor in enumerate(NEIGHBORHOOD):
                target = np.array([patch_row + neighbor[0], patch_col + neighbor[1]])
                print(target, "..........................................")
                
                idx = np.where(np.all(row_col_array == target, axis=1))[0]
                if len(idx) > 0:
                    sample = rc[idx[0]]["sample"]
                    print(sample, ".....................................................")
                    raw_path = f"{s}/raw/{sample}"
                    if raw_path in zarr_f:
                        raw = zarr_f[raw_path][:]
                        r, c = np.unravel_index(i, (3, 3))
                        patch[r * PATCH_SIZE:(r + 1) * PATCH_SIZE, c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = raw
        planes.append(patch)
        image.imsave(f"pics/slice{s}_rows{patch_row - 1}_{patch_row + 1}_cols{patch_col - 1}_{patch_col + 1}.png", patch)

    # Compute phase cross-correlation shifts
    for i in range(1, len(planes)):
        print(np.min(planes[i]), np.max(planes[i]))
        shift, _, _ = phase_cross_correlation(planes[0], planes[i])
        shifts.append(shift)
    
    return shifts

# Parallelized sample collection
def collect_all_samples():
    with Pool(processes=cpu_count()) as pool:
        return pool.map(collect_samples_for_slice, SLICES)

# Parallelized shift computation
def process_shifts_and_save(rc_list):
    with Pool(processes=cpu_count()) as pool:
        return pool.map(process_shift_and_save_image, [(id, rc_list) for id in vastSubs_range])


if __name__ == "__main__":
    rc_list = collect_all_samples()
    shiftZ = process_shifts_and_save(rc_list)

    savemat('shiftZ.mat', {'shiftZ': shiftZ})
