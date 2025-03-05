
import os.path
import gunpowder as gp
import h5py
import io
import logging
import math
import numpy as np
import random
import requests
import torch
from torch import nn
import zarr
from gunpowder.torch import Train
from torch.utils.tensorboard import SummaryWriter
from segmentation_models_pytorch import UnetPlusPlus
import os


input_size = gp.Coordinate((512, 512))
output_size = gp.Coordinate((512, 512))
voxel_size = gp.Coordinate((1, 1))
num_samples=587
batch_size=6
os.environ["TORCH_HOME"] = "./trainingFiles"


class detectionLoss(torch.nn.Module):

    def __init__(self, smooth=1.0):
        super(detectionLoss, self).__init__()
        self.classification_loss = torch.nn.BCELoss()  
        self.regression_loss = torch.nn.SmoothL1Loss()
        self.bce_loss = torch.nn.BCELoss()
        self.smooth=smooth

    def forward(self, preds, targets):
        targets = targets.float()
        intersection = (preds * targets).sum(dim=(1, 2))
        union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        bce_loss = self.bce_loss(preds, targets)
        # loss =  dice_loss + bce_loss

        # print(f"BCE Loss: {bce_loss.item():.4f}, Dice Loss: {dice_loss.item():.4f}, Total Loss: {loss.item():.4f}")
        print(f"Total Loss: {bce_loss.item():.4f}")

        return bce_loss


class detectionModel(torch.nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.unetplusplus = UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=self.in_channels,
            classes=1,
            activation="sigmoid"
        )

    def forward(self, input):
        cls_logits = self.unetplusplus(input)
        cls_logits = torch.squeeze(cls_logits, dim=1)

        return cls_logits


def train(
        checkpoint_name,
        dir,
        max_iteration,
        save_every=100,
        latest_checkpoint=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = detectionModel().to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(model)

    loss = detectionLoss().to(device)
    optimizer = torch.optim.Adam(lr=0.7e-4, params=model.parameters(), weight_decay=0.7e-4)

    if latest_checkpoint is not None:
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ck_filename = os.path.join(dir, "checkpoints")
    if not os.path.exists(ck_filename):
        os.makedirs(ck_filename)

    raw = gp.ArrayKey('RAW')  
    pred = gp.ArrayKey('PRED')
    bbox = gp.ArrayKey('BBOX')

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(bbox, output_size)
    request.add(pred, output_size)

    sources = tuple(
        gp.ZarrSource(
            "datasets/RgcDetection/mip0-size512/mip0-size512.zarr",
            {
                raw: f'raw/{i}',
                bbox: f'bbox_mask/{i}'
            },
            {
                raw: gp.ArraySpec(interpolatable=True, voxel_size=voxel_size),
                bbox: gp.ArraySpec(interpolatable=False, voxel_size=voxel_size),
            }) +
        gp.Pad(raw, size=None) +
        gp.Pad(bbox, size=None) 
        for i in range(num_samples)
    )

    pipeline = sources

    pipeline += gp.RandomProvider()

    pipeline += gp.SimpleAugment()

    pipeline += gp.ElasticAugment(
        control_point_spacing=(40, 40),
        jitter_sigma=(1, 1),
        rotation_interval=(0, math.pi/4))
   
    pipeline += gp.Unsqueeze([raw])

    pipeline += gp.Stack(batch_size)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        checkpoint_basename=os.path.join(dir,'checkpoints',checkpoint_name),
        save_every=save_every,
        log_dir = os.path.join(dir,'logs'),
        
        log_every = save_every, 
        inputs={
            'input': raw,
        },
        outputs={
            0: pred
        },
        loss_inputs={
            0: pred,
            1: bbox,
        })

    with gp.build(pipeline):
        for i in range(max_iteration):
            print(f"training in iteration {i+1} ...")
            summary_writer = SummaryWriter(log_dir=os.path.join(dir,'logs')),
            batch = pipeline.request_batch(request)
    summary_writer.close()


train(
    max_iteration=60000,
    latest_checkpoint="trainingFiles/unetplusplus_RGCdetection/checkpoints/model_ns523_256by256_rgcDet_round3_checkpoint_5100",
    checkpoint_name='model_ns587_512by512_rgcDet_round4',
    dir='./trainingFiles/unetplusplus_RGCdetection',
    save_every=100
)
