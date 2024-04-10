import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import argparse

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


def main(record_number):
    BACKBONE = 'resnet34'

    # custom_objects = {
    #     'binary_focal_crossentropy': tf.keras.losses.BinaryFocalCrossentropy(
    #         apply_class_balancing=True, alpha=0.9, gamma=2.0, from_logits=False),
    #     'iou_score': sm.metrics.iou_score
    # }

    # Load pre-trained model with torch
    model = torch.load("bestmodel4.pth")  # Does the model need to be updated for PyTorch?
    
    dataset_dir = "/test_dataset" # Update the dataset directory
    val_images, val_masks = [], []

    # TODO: 
    # for num in range(record_number):
    #     path = f"{dataset_dir}/validation.tfrecords-{num:05}-of-00100"
    #     dataset_val = tf.data.TFRecordDataset(path).map(parse_example)
        
    #     for feature in dataset_val.as_numpy_iterator():
    #         if np.any(feature['human_pixel_masks'] == 1):
    #             n_times_before = feature['n_times_before']
    #             val_images.append(false_color_image(feature)[..., n_times_before, :])
    #             val_masks.append(tf.squeeze(feature['human_pixel_masks'], axis=-1))

    val_images = torch.tensor(val_images)
    val_masks = torch.tensor(val_masks)
    
    # Set model to eval mode
    model.eval()

    # review https://stackoverflow.com/questions/73396203/how-to-use-trained-pytorch-model-for-prediction
    with torch.no_grad():
        predicted_masks_val = (model(val_images) > 0.5).int()

    # .flatten is numpy, so no need to change
    f1_scores = [f1_score(true_mask.flatten().cpu().numpy(), pred_mask.flatten().cpu().numpy(), average='binary')
                 for true_mask, pred_mask in zip(val_masks, predicted_masks_val)]
    
    return np.mean(f1_scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("record_number", type=int, help="Number of validation datasets to process")
    args = parser.parse_args()
    print(main(args.record_number))
