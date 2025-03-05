from multiprocessing import Pool, cpu_count
from matplotlib import image
import zarr
import numpy as np
import os
from skimage.measure import label, regionprops
from scipy.io import savemat

def process_sample(sample, slice_num, zarr_f, slice_name=None):
    global diced_size, rowRange, colRange, threshold
    
    if slice_name is not None:
       row_col = zarr_f[f"{slice_name}/xy_coordinates/{sample}"]
    else:
       row_col = zarr_f[f"xy_coordinates/{sample}"]
    row, col = int(row_col[0]), int(row_col[1])
    
    if row in rowRange and col in colRange:
        pred = zarr_f[f"pred/{sample}"]
        pred = (np.array(pred) > threshold).astype("int")
        
        comps = label(pred, connectivity=1)
        props = regionprops(comps)
        
        vsub_list = []
        for prop in props:
             vsub = []
             idxs = np.where(comps == prop.label)
             for i in range(len(idxs[0])):
                 coors = [(row * diced_size[0] + idxs[0][i]) // 16, (col * diced_size[1] + idxs[1][i]) // 16, slice_num]
                 vsub.append(coors)
             vsub = np.array(vsub)
             vsub = np.unique(vsub, axis=0)
             if len(vsub) > 100:
                vsub_list.append(vsub)
        
        png_filename = f"{row}_{col}.png"
        print(f"{png_filename} saved.")
        
        return vsub_list
    return []

if __name__ == "__main__":
    root_dir = "/storage1/fs1/jlmorgan/Active/morganLab/DATA/LGN_Developing/KxR_P11LGN/diced"
    zarr_name_pattern = "predict_RGCdetection_plane*_checkpoint3900.zarr"
    chk = zarr_name_pattern.split("_checkpoint")[-1].split(".zarr")[0]
    slice_nums = range(749, 760)
    threshold = 0.2
    diced_size = [512, 512]
    startRow, stopRow = 0, 200
    startCol, stopCol = 0, 200
    rowRange = range(startRow, stopRow + 1)
    colRange = range(startCol, stopCol + 1)
    all_slices_vastSubs = []
    for slice_num in slice_nums:
        zarr_filename = os.path.join(root_dir, zarr_name_pattern.replace("*", str(slice_num + 1)))
        zarr_f = zarr.open(zarr_filename, 'r')
        slice_name = None # "750"
        sample_path = os.path.join(zarr_filename, "raw")
        samples = [item for item in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, item))]
        args = [(sample, slice_num, zarr_f, slice_name) for sample in samples]

        with Pool(processes=cpu_count()) as pool:
            vastSubs = pool.starmap(process_sample, args)
    
        vastSubs = [vsub for vsub_list in vastSubs for vsub in vsub_list]
        if isinstance(vastSubs, np.ndarray):
           vastSubs = vastSubs.tolist()
        all_slices_vastSubs.append(vastSubs)
        print(f"vastSubs for plane {slice_num} is calculated.......")

    save_filename = os.path.join(root_dir, f'vastSubs_planes{slice_nums}_thr{threshold}_chk{chk}.mat')
    savemat(save_filename, {'vastSubs': all_slices_vastSubs})
    print(f"vastSubs for planes {slice_nums} was saved in {save_filename}.")

