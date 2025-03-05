from multiprocessing import Pool, cpu_count
from matplotlib import image
import zarr
import numpy as np
import os
from skimage.measure import label, regionprops
from scipy.io import savemat


def process_sample(sample, slice_name, zarr_f):
    global diced_size, rowRange, colRange, mito_probability_threshold, pred_key_name, mipLevel, voxel_numbers_threshold

    ds_factor = 2 ** mipLevel
    row_col = sample.split("_")
    row, col = int(row_col[0]), int(row_col[1])
    slice_num = int(slice_name) - 1
    
    if row in rowRange and col in colRange:
        pred = zarr_f[f"{slice_name}/{pred_key_name}/{sample}"]
        pred = (np.array(pred) > mito_probability_threshold).astype("int")
        
        comps = label(pred, connectivity=1)
        props = regionprops(comps)
        
        vsub_list = []
        for prop in props:
            idxs = np.where(comps == prop.label)
            vsub = np.column_stack([
                (row * diced_size[0] + idxs[0]) // ds_factor,
                (col * diced_size[1] + idxs[1]) // ds_factor,
                np.full_like(idxs[0], slice_num)
            ])
            vsub = np.unique(vsub, axis=0)
            if len(vsub) > voxel_numbers_threshold:
                vsub_list.append(vsub)
                
        print(f"{sample} in plane {slice_name} completed.")
        
        return vsub_list
    return []

if __name__ == "__main__":
    root_dir = "/storage1/fs1/jlmorgan/Active/morganLab/DATA/LGN_Developing/KxR_P11LGN/diced"
    zarr_name = "predict_RGCdetection_planes_range(78, 86)_checkpoint3900.zarr"
    zarr_filename = os.path.join(root_dir, zarr_name)
    save_name = 'vastSubs.mat'
    save_filename = os.path.join(root_dir, save_name)
    zarr_f = zarr.open(zarr_filename, 'r')
    slice_names = list(zarr_f.keys())
    mito_probability_threshold = 0.4
    voxel_numbers_threshold = 100
    pred_key_name = "pred"
    diced_size = [512, 512]
    startRow, stopRow = 0, 200
    startCol, stopCol = 0, 200
    mipLevel = 4
    rowRange = range(startRow, stopRow + 1)
    colRange = range(startCol, stopCol + 1)
    all_slices_vastSubs = []
    for slice_name in slice_names:
        slice_keys = list(zarr_f[f"{slice_name}/raw"].keys())
        args = [(sample, slice_name, zarr_f) for sample in slice_keys]

        with Pool(processes=cpu_count()) as pool:
            vastSubs = pool.starmap(process_sample, args)
    
        vastSubs = [vsub for vsub_list in vastSubs for vsub in vsub_list]
        if isinstance(vastSubs, np.ndarray):
           vastSubs = vastSubs.tolist()
        all_slices_vastSubs.append(vastSubs)
        print(f"vastSubs for plane {slice_name} is calculated.......")

    savemat(save_filename, {'vastSubs': all_slices_vastSubs})
    print(f"vastSubs for planes {slice_names} was saved in {save_filename}.")

