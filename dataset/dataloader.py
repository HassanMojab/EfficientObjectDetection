import pandas as pd
import numpy as np
import warnings

from torch.utils.data.dataset import Dataset
from PIL import Image

from utils import utils
from constants import num_actions

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transform, num_act=num_actions):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        data_info = pd.read_csv(csv_path, header=None)
        # Second column is the image paths
        self.image_arr = np.asarray(data_info.iloc[:, 1])
        # First column is the image IDs
        self.label_arr = np.asarray(data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(data_info)

        # Preload metrics
        self.offset_fd, self.offset_cd = utils.read_offsets(self.label_arr, num_act)
        self.object_counts = utils.read_counts(self.label_arr, num_act)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        offset_fd = self.offset_fd[index]
        offset_cd = self.offset_cd[index]
        object_counts = self.object_counts[index]

        return (img_as_tensor, single_image_label, offset_fd, offset_cd, object_counts)

    def __len__(self):
        return self.data_len
