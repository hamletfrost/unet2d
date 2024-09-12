import numpy as np
import imgaug as ia
import torch.utils as utils

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

class AugTransform:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x, y_true):
        # normalize
        x = normalize(x)

        # augment
        y_true = ia.SegmentationMapsOnImage(y_true, shape=y_true.shape)
        x_aug, y_true_aug = self.aug(image=x, segmentation_maps=y_true)

        # reshape
        x_aug = np.expand_dims(x_aug, 0).astype(np.float32)
        y_true_aug = np.expand_dims(y_true_aug.arr[:, :, 0], 0).astype(np.float32)

        return {"x": x_aug, "y_true": y_true_aug}

# Note: tfelt additions are reverse compatible
#       they just allow for custom weights (func on init)
#       and also normalize before augmentation so contrast augs work
#        (even tho test data is also normalized, better to learn a wider range)
class AugTransformConstantPosWeights:
    """creates weights with foreground to 1 * pos_weight and background to 1"""
    def __init__(self, aug, pos_weight, weights_func=None):
        self.aug = aug
        self.pos_weight = pos_weight
        if weights_func is None:          # tfelt addition
            self.w_func = default_weights #
        else:                             #
            self.w_func = weights_func    #

    # tfelt addition
    def default_weights(w, x, y):
        weights = w * (y_true_aug > 0).astype(np.float32) +1
        return weights

    def __call__(self, x, y_true):
        # normalize
        x = normalize(x)

        # augment
        y_true = ia.SegmentationMapsOnImage(y_true, shape=y_true.shape)
        x_aug, y_true_aug = self.aug(image=x, segmentation_maps=y_true)

        # reshape
        x_aug = np.expand_dims(x_aug, 0).astype(np.float32)
        y_true_aug = np.expand_dims(y_true_aug.arr[:, :, 0], 0).astype(np.float32)
        #weights = self.pos_weight * (y_true_aug > 0).astype(np.float32) + 1
        weights = self.w_func(self.pos_weight, x_aug, y_true_aug) # tfelt addition

        return {"x": x_aug, "y_true": y_true_aug, "weights": weights}

class AugTransformMultiClass:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, x, y_true):
        # normalize
        x = normalize(x)

        # augment
        y_true = ia.SegmentationMapsOnImage(y_true, shape=y_true.shape)
        x_aug, y_true_aug = self.aug(image=x, segmentation_maps=y_true)

        # reshape
        x_aug = np.expand_dims(x_aug, 0).astype(np.float32)
        y_true_aug = np.expand_dims(y_true_aug.arr[:, :, 0], 0).astype(np.long)
        weights = (y_true_aug >= 0).astype(np.float32)

        return {"x": x_aug, "y_true": y_true_aug, "weights": weights}

class Dataset(utils.data.Dataset):
    def __init__(self, x_array, y_true_array, transform):
        self.x_array = x_array
        self.y_true_array = y_true_array
        self.data_len = x_array.shape[0]
        self.transform = transform

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        x = self.x_array[idx, :, :]
        y_true = self.y_true_array[idx, :, :]
        sample = self.transform(x, y_true)

        return sample
