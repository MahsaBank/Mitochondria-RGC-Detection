This repository contains the code for training an RGC detector model (train.py script) and predicting mitochondria (inference.py) in RGC boutons within an EM volume using the trained detector. The RGC detector is a U-Net++ model with a ResNet-34 encoder, designed to predict the probability of each pixel belonging to an RGC mitochondrion. These probabilities can be thresholded to obtain segmented RGC mitochondria. Finally, the segmented 3D mitochondria can be rendered using the "predictionsToVastSubs.py" script. You can find snapshots of the segments in the "test_results" folder within this repository.

The dataset is stored in Zarr format with two key names: "raw" and "pred". The EM stack is organized into plane folders, with each plane folder containing diced patches of raw images arranged in a "row_col" structure. Each patch is a 512 × 512 array. After running inference.py, the predictions are saved under the "pred" entry in the Zarr file.


