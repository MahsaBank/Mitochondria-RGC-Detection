import os
import glob
import zarr
import numpy as np
import scipy.io as sio
from scipy.io import savemat
from skimage.registration import phase_cross_correlation
from multiprocessing import Pool, cpu_count
from matplotlib import image


ZARR_FILENAME = "/storage1/fs1/jlmorgan/Active/morganLab/DATA/LGN_Developing/KxR_P11LGN/diced/KxR_P11LGN_planes_range(1, 39)_mip0.zarr"
ROWS = range(0, 200, 3)
COLS = range(0, 100, 3)
depth_size = 10
SLICES = range(35, 39, depth_size)
zarr_f = zarr.open(ZARR_FILENAME, 'a')


def calculate_shift(chunk_data):
    plane_start_id, row_col = chunk_data
    shift_key_name = f"{plane_start_id}/shift/{row_col}"

    if shift_key_name in zarr_f:
        print(f"{shift_key_name} is already available ....")
        return

    row_start_id, col_start_id = map(int, row_col.split('_'))

    planes = []
    shifts = []
    correlations = []

    for s in range(plane_start_id, plane_start_id + depth_size):
        patch = np.zeros((3 * 512, 3 * 512))

        for r in range(row_start_id, row_start_id + 3):
            for c in range(col_start_id, col_start_id + 3):
                entry_key_name = f"{s}/raw/{r}_{c}"

                if entry_key_name in zarr_f:
                    patch[(r - row_start_id) * 512: (r - row_start_id + 1) * 512,
                          (c - col_start_id) * 512: (c - col_start_id + 1) * 512] = np.asarray(zarr_f[entry_key_name])

        planes.append(patch)

    if len(planes) < 2:
        print(f"Insufficient planes found for {shift_key_name}.")
        return

    # Compute phase cross-correlation shifts
    for i in range(1, len(planes)):
        shift, error, _ = phase_cross_correlation(planes[0], planes[i])
        correlations.append(1 - error)
        shifts.append(shift)

    best_idx = np.argmax(correlations)
    best_shift = shifts[best_idx]
    zarr_f[shift_key_name] = best_shift
    print(f"{shift_key_name} was created ....")


if __name__ == "__main__":
    all_chunk_data = [(s, f"{r}_{c}") for s in SLICES for r in ROWS for c in COLS]

    with Pool(processes=cpu_count()) as pool:
        pool.map(calculate_shift, all_chunk_data)
