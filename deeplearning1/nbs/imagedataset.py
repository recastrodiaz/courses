import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import glob
import numpy as np
import torch

def open_image(path):
    img = Image.open(path).convert('RGB')
    # Center crop 
    width = img.size[0]
    height = img.size[1]
    return img.crop(
        (
            (width  - 224) / 2,
            (height - 224) / 2,
            (width  + 224) / 2,
            (height + 224) / 2
        )
    )

def get_dataframe_row_by_id(df, id, test_mode):
    if test_mode:
        # TODO can we do better than this?
        return 0
    # Return a DataFrame row by its id in numpy format
    row = df.loc[id].as_matrix().astype(np.float32)
    return torch.FloatTensor(row)

def get_image_id(file_path):
    # Remove everything before the last /
    filename = file_path.rsplit('/',1)[1]
    # Remove everything before the last '.'
    file_id = filename.rsplit('.',1)[0]
    return int(file_id)

def make_dataset(dir, target_dataframes, test_mode):
    # Read all image files in directory
    images_paths = glob.glob(dir +"/*.jpg")
    
    # TODO
    if test_mode:
        print("Using test mode")
    
    # Create an array of tuples (tensor image and its target vector), one row per image
    images_targets = [ 
        (open_image(image_path), # image
         get_dataframe_row_by_id(target_dataframes, get_image_id(image_path), test_mode)) # target
        for image_path in images_paths]
        
    return images_targets, images_paths

def get_images_ids_from_image_paths(images_paths):
    return [get_image_id(image_path) for image_path in images_paths]

class ImageDataset(data.Dataset):

    def __init__(self, root, target_dataframes, test_mode):
        images_targets, images_paths = make_dataset(root, target_dataframes, test_mode)
        if len(images_targets) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root ))
        
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.root = root
        self.images_targets = images_targets
        self.images_idx_to_id = get_images_ids_from_image_paths(images_paths)
        # images_idx_to_id[5] --> 145689 (.jpg)

    def __getitem__(self, index):
        image, target = self.images_targets[index]
        tensor = self.image_transform(image)
        return tensor, target

    def __len__(self):
        return len(self.images_targets)