import os
import glob
import zarr
import numpy as np
import scipy.io as sio
from scipy.io import savemat
from skimage.registration import phase_cross_correlation
from multiprocessing import Pool
from matplotlib import image

# File paths
ZARR_FILENAME = "/storage1/fs1/jlmorgan/Active/morganLab/DATA/LGN_Developing/KxR_P11LGN/diced/KxR_P11LGN_mip0_s750_799.zarr"
VASTSUBS_FILENAME = "/storage1/fs1/jlmorgan/Active/morganLab/DATA/LGN_Developing/KxR_P11LGN/CellNav_KxR/vastSubsPlanes749-759.mat"

# Parameters
NEIGHBORHOOD = [
    [-1, -1], [-1, 0], [-1, 1], 
    [0, -1], [0, 0], [0, 1], 
    [1, -1], [1, 0], [1, 1]
]
SLICES = range(750, 761)

# Load data
vastSubs = sio.loadmat(VASTSUBS_FILENAME)["vastSubs"][0]
vastSubs_range = range(len(vastSubs))
zarr_f = zarr.open(ZARR_FILENAME, 'r')

# Helper function to collect samples for each slice
def collect_samples_for_slice(slice_num):
    sample_paths = glob.glob(os.path.join(ZARR_FILENAME, f"{slice_num}/raw/*"))
    samples = sorted([int(os.path.basename(sample)) for sample in sample_paths if os.path.exists(sample)])

    rc = []
    for sample in samples:
        row_col_path = f"{slice_num}/xy_coordinates/{sample}"
        if row_col_path in zarr_f:
            row_col = zarr_f[row_col_path][:]
            rc.append({"sample": sample, "rowCol": np.array(row_col).astype(int)})

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
    for s in range(slice_num, slice_num + 10):

        patch = np.zeros((3 * 512, 3 * 512))
        if rc:
            min_row, min_col = rc[0]["rowCol"]
            patch_row = (min_row * 16) // 512
            patch_col = (min_col * 16) // 512

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
                        patch[r * 512:(r + 1) * 512, c * 512:(c + 1) * 512] = raw
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
    with Pool(processes=4) as pool:
        return pool.map(process_shift_and_save_image, [(id, rc_list) for id in vastSubs_range])

# Main execution
if __name__ == "__main__":
    rc_list = collect_all_samples()
    shiftZ = process_shifts_and_save(rc_list)

    # Save results
    savemat('shiftZ.mat', {'shiftZ': shiftZ})
