import os
import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningDataModule, Trainer
from sklearn.metrics import f1_score
import argparse
import segmentation_models_pytorch as smp
import ssl
from contrail import ContrailModel
from image_dataset import ImageDataset, visualize
from loss import DiceLoss, FocalLoss
ssl._create_default_https_context=ssl._create_unverified_context

_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)

def parse_example(serialized_example: bytes) -> dict:
    goes_bands = ['upper_vapor', 'mid_vapor', 'low_vapor', 'cloud_top',
                  'ozone', 'data_10um', 'data_11um', 'data_12um', 'co2']
    feature_spec = {}
    for band in goes_bands:
        feature_spec.add()
        feature_spec[band] = torch.empty((), dtype=torch.str)
    feature_spec.update({
        # data_11um - data_12um
        'brightness_temperature_difference': torch.Tensor(dtype=torch.str),
        'human_pixel_masks': torch.Tensor(dtype=torch.str),
        'n_times_before': torch.Tensor(dtype=torch.int64),
        'n_times_after': torch.Tensor(dtype=torch.int64),
        
        # Projection params
        'projection_wkt': torch.Tensor(dtype=torch.str),
        'col_min': torch.Tensor(dtype=torch.float32),
        'row_min': torch.Tensor(dtype=torch.float32),
        'col_size': torch.Tensor(dtype=torch.float32),
        'row_size': torch.Tensor(dtype=torch.float32),
        
        # Timestamp 
        'timestamp': torch.Tensor(dtype=torch.int64),

        # should have variable length but there is no direct equivalent in pytorch, may have to adjust
        'satellite_scan_starts': torch.Tensor(dtype=torch.int64),
    })

    
    # not sure if this conversion is correct
    features = serialized_example
    for key in [*goes_bands, 'brightness_temperature_difference']:
        features[key] = torch.tensor(features[key], dtype=torch.double)
        features['human_pixel_masks'] = torch.tensor(features['human_pixel_masks'], dtype=torch.int32)
    return features

def normalize_range(data, bounds):
    """
    Map data to the range [0, 1]
    """

    return (data - bounds[0]) / (bounds[1] - bounds[0])


def false_color_image(brightness_temperatures):
    """
    Generate ash false color image from GOES brightness temperatures
    """

    r = normalize_range(
        brightness_temperatures['data_12um'] -
        brightness_temperatures['data_11um'], _TDIFF_BOUNDS)
    g = normalize_range(
        brightness_temperatures['data_11um'] -
        brightness_temperatures['cloud_top'], _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(brightness_temperatures['data_11um'], _T11_BOUNDS)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 1)


def main(image_path, mask_path):
    # BACKBONE = 'resnet152'

    # Load pre-trained model with torch

    # loaded_model = smp.create_model(arch="Unet", encoder_name="resnet152", in_channels=3, classes=1)
    # model.load_state_dict(torch.load("models/contrailsonly.torch.states.0.1.ResNet152.Dice.bin"), strict=False)
    # model.eval()

    loss = DiceLoss(log_loss=True)
    # weights = torch.load("models/contrailsonly.torch.states.0.1.ResNet152.Dice.bin", weights_only=True) # added line
    model = ContrailModel(
            arch="Unet",
            encoder="resnet152",
            # pretrained = weights, # added line
            pretrained="imagenet",
            in_channels=3,
            classes=1,
            loss_function=loss,
            learning_rate=0.001,
        )

    model.load_state_dict(torch.load("models/contrailsonly.torch.states.0.1.ResNet152.Dice.bin"), strict=False)

    val_image = np.load(image_path).transpose(2, 0, 1).astype(np.float32) 
    val_mask = np.load(mask_path)
    # val_mask = val_mask / val_mask.max() # normalizing mask
    val_mask = val_mask[..., np.newaxis].transpose(2, 0, 1) # 1, 256, 256
    print(f"image[{val_image.ndim}]: {val_image.shape}, mask[{val_mask.ndim}]: {val_mask.shape}")

    val_image = val_image[..., np.newaxis].transpose(3, 0, 1, 2)
    print(f"image_dim[{val_image.ndim}]: {val_image.shape}, mask_dim[{val_mask.ndim}]: {val_mask.shape}")
    
    
    val_image = torch.from_numpy(val_image)
    
    model.eval()
    with torch.inference_mode(): 
        y_hat = model(val_image).sigmoid().squeeze()
        print(f"[{y_hat.ndim}]: {y_hat.shape}")

    # at this point y_hat has shape: torch.Size([256, 256]), and val_mask has shape:(1, 256, 256)
    # visualize(y_hat=y_hat, y=val_mask)


    compare_image = np.zeros_like(val_mask)

    for i in range(y_hat.shape[0]):
        for j in range(y_hat.shape[1]):
            if val_mask[0, i, j] == 1 and y_hat[i, j] >= 0.5:  # True Positive
                compare_image[0, i, j] = int("CCFF00",16) 
            elif val_mask[0, i, j] == 0 and y_hat[i, j] < 0.5:  # True Negative
                compare_image[0, i, j] = int("FFFFFF",16) 
            elif val_mask[0, i, j] == 0 and y_hat[i, j] >= 0.5:  # False Positive
                compare_image[0, i, j] = int("F4BBFF",16) 
            elif val_mask[0, i, j] == 1 and y_hat[i, j] < 0.5:  # False Negative
                compare_image[0, i, j] = int("9BDDFF",16)

    visualize(y_hat=y_hat, compare_image=compare_image, y=val_mask)

    return y_hat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", dest="image_path", type=str, help="Path to image", required=False, default="test_dataset/singleframeimage/image-val-ts1557132000-n00011.npy")
    parser.add_argument("--mask", dest="mask_path", type=str, help="Path to image", required=False, default="test_dataset/mask/mask-val-ts1557132000-n00011.npy")
    args = parser.parse_args()
    # print(main(args.image_path))
    # print(args.image_path, args.mask_path)
    print(main(args.image_path, args.mask_path))
