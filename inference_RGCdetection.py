import gunpowder as gp
import numpy as np
import matplotlib.pyplot as plt
import zarr
import torch
from torch import nn
from funlib.learn.torch.models import UNet, ConvPass
import os
import torch.nn.functional as F
from segmentation_models_pytorch import UnetPlusPlus


num_fmaps = 16
input_size =  gp.Coordinate((512, 512))
output_size = gp.Coordinate((512, 512))
padding_size = ((0, 0), (0, 0)) #(256-248)/2

class detectionModel(torch.nn.Module):

    def __init__(self, in_channels=1, num_fmaps=num_fmaps, bbox_num=10):
        super().__init__()
        self.in_channels = in_channels
        self.bbox_num = bbox_num

        self.unetplusplus = UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation="sigmoid"
        )

        self.classification_head = ConvPass(num_fmaps, 1, [[1, 1]], activation='Sigmoid')

    def forward(self, input):

        cls_logits = self.unetplusplus(input)
        cls_logits = torch.squeeze(cls_logits, dim=1)

        return cls_logits


#@title utility function to view labels

# matplotlib uses a default shader
# we need to recolor as unique objects

def create_lut(labels):

    max_label = np.max(labels)

    lut = np.random.randint(
            low=0,
            high=255,
            size=(int(max_label + 1), 3),
            dtype=np.uint64)

    lut = np.append(
            lut,
            np.zeros(
                (int(max_label + 1), 1),
                dtype=np.uint8) + 255,
            axis=1)

    lut[0] = 0
    colored_labels = lut[labels]

    return colored_labels


def predict(checkpoint, raw_file, plane, raw_idx):

    raw = gp.ArrayKey('RAW')
    pred = gp.ArrayKey('PRED')

    scan_request = gp.BatchRequest()
    scan_request.add(raw, input_size)
    scan_request.add(pred, output_size)

    context = (input_size - output_size)

    source = gp.ZarrSource(
        raw_file,
        {
            raw: f'{plane}/raw/{raw_idx}',
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
        }
    )

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = source.spec[raw].roi.grow(-context, -context)

    model = detectionModel()
    model.eval()

    predict = gp.torch.Predict(
        model=model,
        checkpoint=checkpoint,
        inputs={
            'input': raw,
        },
        outputs={
            0: pred,
        }
    )

    scan = gp.Scan(scan_request)

    pipeline = source

    # raw shape = h,w
    pipeline += gp.Unsqueeze([raw])
    # raw shape = c,h,w

    pipeline += gp.Stack(1)
    # raw shape = b,c,h,w

    pipeline += predict

    pipeline += scan

    pipeline += gp.Squeeze([raw])
    # raw shape = c,h,w

    pipeline += gp.Squeeze([raw, pred])
    # raw shape = h,w

    predict_request = gp.BatchRequest()

    predict_request.add(raw, total_input_roi.get_end())
    predict_request.add(pred, total_output_roi.get_end())

    with gp.build(pipeline):
        batch = pipeline.request_batch(predict_request)

    return batch[raw].data, batch[pred].data


checkpoints = ["trainingFiles/unetplusplus_RGCdetection/checkpoints/model_ns587_512by512_rgcDet_round4_checkpoint_3900"]
raw_file = "/storage1/fs1/jlmorgan/Active/morganLab/DATA/LGN_Developing/KxR_P11LGN/diced/KxR_P11LGN_mip0_s001_749.zarr"
planes = range(102, 749)
start_idx = np.zeros((1, len(planes))) 
start_idx[0] = 6028
save_pngs = []

base_path = os.path.dirname(raw_file)
base_name = os.path.basename(raw_file)
zarr_name = base_name
png_name = base_name.split(".zarr")[0]
zarr_f = zarr.open(raw_file, mode='a')

for checkpoint in checkpoints:

    chk_num = checkpoint.split("checkpoint_")[-1]
    #pred_data_filename = f'{base_path}/predict_RGCdetection_planes_{planes}_checkpoint{chk_num}.zarr'
    #fi = zarr.open(pred_data_filename, mode='a')

    for i, plane in enumerate(planes):

        plane_path = os.path.join(raw_file, str(plane), "raw")
        if os.path.exists(plane_path):
           raw_idx = [item for item in os.listdir(plane_path) if os.path.isdir(os.path.join(plane_path, item))]
        else:
           raw_idx = [] 

        for idx in raw_idx:             
            raw, pred = predict(checkpoint, raw_file, plane, idx)

            pred = np.pad(pred, padding_size)

            zarr_f[f'{plane}/pred/{idx}'] = pred

            if idx in save_pngs:

                save_filename = f'{base_path}/predict_RGCdetection_plane{plane}_img{idx}.png'

                fig, axes = plt.subplots(1, 1)

                axes.imshow(raw, cmap='gray')
                axes.imshow(pred, cmap='jet', alpha=0.3)

                plt.savefig(save_filename)
