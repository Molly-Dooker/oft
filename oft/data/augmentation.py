import math
import random
from PIL import ImageOps
import torch
from torch.utils.data import Dataset
from .. import utils
from torchvision import transforms
import ipdb
import albumentations as A # Albumentations 임포트
from albumentations.pytorch import ToTensorV2 # Albumentations의 텐서 변환기
import numpy as np
def random_crop(image, calib, objects, output_size):
    # Randomize bounding box coordinates
    width, height = image.size
    w_out, h_out = output_size
    left = random.randint(0, max(width - w_out, 0))
    upper = random.randint(0, max(height - h_out, 0))
    # Crop image
    right = left + w_out
    lower = upper + h_out
    image = image.crop((left, upper, right, lower))
    # Modify calibration matrix
    calib[0, 2] = calib[0, 2] - left                # cx' = cx - du
    calib[1, 2] = calib[1, 2] - upper               # cy' = cy - dv
    calib[0, 3] = calib[0, 3] - left * calib[2, 3]  # tx' = tx - du * tz
    calib[1, 3] = calib[1, 3] - upper * calib[2, 3] # ty' = ty - dv * tz
    # Include only visible objects
    cropped_objects = list()
    for obj in objects:
        pos_2d = utils.perspective(calib, calib.new(obj.position))
        upos, vpos = pos_2d[0], pos_2d[1]
        is_visible = (upos >= 0) and (upos < w_out) and (vpos >= 0) and (vpos < h_out)        
        if is_visible: cropped_objects.append(obj)
    return image, calib, cropped_objects

def random_scale(image, calib, scale_range=(0.8, 1.2)):
    scale = random.uniform(*scale_range)
    # Scale image
    width, height = image.size
    image = image.resize((int(width * scale), int(height * scale)))
    # Scale first two rows of calibration matrix
    calib[:2, :] *= scale
    return image, calib


def random_flip(image, calib, objects):
    if random.random() < 0.5: return image, calib, objects
    # Flip image
    image = ImageOps.mirror(image)
    # Modify calibration matrix
    width, _ = image.size
    calib[0, 2] = width - calib[0, 2]               # cx' = w - cx
    calib[0, 3] = width * calib[2, 3] - calib[0, 3] # tx' = w*tz - tx
    # Flip object x-positions
    flipped_objects = list()
    for obj in objects:
        position = [-obj.position[0]] + obj.position[1:]
        angle = math.atan2(math.sin(obj.angle), -math.cos(obj.angle))
        flipped_objects.append(utils.ObjectData(
            obj.classname, position, obj.dimensions, angle, obj.score
        ))
    return image, calib, flipped_objects


def random_crop_grid(grid, objects, crop_size):
    # Get input and output dimensions
    grid_d, grid_w, _ = grid.size()
    crop_w, crop_d = crop_size
    # Try and find a crop that includes at least one object
    for _ in range(10): # Timeout after 10 attempts
        # Randomize offsets
        xoff = random.randrange(grid_w - crop_w) if crop_w < grid_w else 0
        zoff = random.randrange(grid_d - crop_d) if crop_d < grid_d else 0
        # Crop grid
        cropped_grid = grid[zoff:zoff+crop_d, xoff:xoff+crop_w].contiguous()
        # If there are no objects present, any random crop will do
        if len(objects) == 0:
            return cropped_grid        
        # Get bounds
        minx, _, minz = cropped_grid.view(-1, 3).min(dim=0)[0]
        maxx, _, maxz = cropped_grid.view(-1, 3).max(dim=0)[0]
        # Search objects to see if any lie within the grid
        for obj in objects:
            objx, _, objz = obj.position
            # If any object overlaps with the grid, return
            if minx < objx < maxx and minz < objz < maxz:
                return cropped_grid
    return cropped_grid
    
def random_jitter_grid(grid, std):
    grid += torch.randn(3) * torch.tensor(std)
    return grid

class AugmentedObjectDataset(Dataset):
    def __init__(self, dataset, image_size=(1080, 360), grid_size=(160, 160), 
                 scale_range=(0.8, 1.2), jitter=[.25, .5, .25]):
        self.dataset = dataset
        self.image_size = image_size
        self.grid_size = grid_size
        self.scale_range = scale_range
        self.jitter = jitter
        self.color_jitter   = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
        # self.random_erasing = transforms.RandomErasing(p=0.5, scale=(0.1, 0.2), ratio=(0.1, 3.3))
        self.dropout = A.Compose([
            A.CoarseDropout(num_holes_range=(1,2),
                            hole_height_range=(0.1,0.15),
                            hole_width_range=(0.1,0.15)),
                            ToTensorV2()])
    def __len__(self):
        return len(self.dataset)    
    def __getitem__(self, index):
        idx, image, calib, objects, grid = self.dataset[index]
        # Apply image augmentation
        image, calib = random_scale(image, calib, self.scale_range)
        image, calib, objects = random_crop(image, calib, objects, self.image_size)
        image, calib, objects = random_flip(image, calib, objects)
        image = self.color_jitter(image)
        image = self.dropout(image=np.array(image))['image']
        image = transforms.ToPILImage()(image)
        # image = transforms.ToPILImage()(image)
        # Augment grid
        grid = random_crop_grid(grid, objects, self.grid_size)
        # grid = random_jitter_grid(grid, self.jitter)
        return idx, image, calib, objects, grid