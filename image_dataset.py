import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from matplotlib.colors import LinearSegmentedColormap

IMG_SIZE = 256


class ImageDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, augmentation=None, preprocessing=None):
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        if self.mask_paths:
            assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self):
        """
        Retrieve the number of samples
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load samples from dataset, preprocess them and return them for a given index
        """
        # image = np.load(self.image_paths[idx]).transpose(2, 0, 1).astype(np.float32)
        # print('LAST IMAGE:')
        # print(self.image_paths[idx])
        image = np.load(self.image_paths[idx]).transpose(2, 3, 0, 1).astype(np.float32)
        # image = np.load(self.image_paths[idx])
        if self.mask_paths:
            mask = np.load(self.mask_paths[idx])
            mask = mask / mask.max()
            mask = mask[..., np.newaxis].transpose(2, 0, 1)
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample["image"], sample["mask"]
            return image, mask
        # no labels
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample["image"]
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]
        return image

def visualize(**images):
    """
    Data visualization
    """
    n = len(images)
    plt.figure(figsize=(n * 5, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.title())
        if image.ndim == 3:
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
        plt.imshow(image.squeeze())
    plt.tight_layout()
    plt.show()

# Create custom colormap for coloring
colors = {
    'TP': '#CCFF00', 
    'TN': '#FFFFFF',  
    'FP': '#F4BBFF',  
    'FN': '#9BDDFF'  
}
custom_colors = LinearSegmentedColormap.from_list("custom_cmap", list(colors.values()))

def visualize_color(**images):
    """
    Data visualization with custom coloring
    """
    n = len(images)
    plt.figure(figsize=(n * 5, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.title())
        if image.ndim == 3:
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
        plt.imshow(image.squeeze(), cmap=custom_colors)
    plt.tight_layout()
    plt.show()


def visualize_batch(batch):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training images")
    plt.imshow(np.transpose(make_grid(batch[0][:64], padding=2, normalize=True), (1, 2, 0)))
    plt.show()


def get_train_augmentation(image_size=IMG_SIZE):
    """
    Augmentation for training images
    """
    transform = [
        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=180, shift_limit=0.3, border_mode=0, p=1),
        albu.PadIfNeeded(min_height=image_size, min_width=image_size, always_apply=True, border_mode=0),
        albu.Resize(image_size, image_size),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
    ]
    return albu.Compose(transform)


def get_val_augmentation(image_size=IMG_SIZE):
    """
    Augmentation for validation images
    """
    transform = [
        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=180, shift_limit=0.3, border_mode=0, p=1),
        albu.PadIfNeeded(min_height=image_size, min_width=image_size, always_apply=True, border_mode=0),
        albu.Resize(image_size, image_size),
    ]
    return albu.Compose(transform)


def get_test_augmentation(image_size=IMG_SIZE):
    """
    Augmentation for testing images (pad and resize only)
    """
    transform = [
        albu.PadIfNeeded(min_height=image_size, min_width=image_size, always_apply=True, border_mode=0),
        albu.Resize(image_size, image_size),
    ]
    return albu.Compose(transform)


def get_preprocessing(encoder):
    """
    Retrive preprocessing transform
    """

    def to_tensor(input, **kwargs):
        return np.expand_dims(input, 0).astype("float64")

    transform = [albu.Lambda(image=to_tensor, mask=to_tensor)]
    return albu.Compose(transform)
