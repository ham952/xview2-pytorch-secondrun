#############################################################################################################
# xView2 Challenge                                                                                          #                                                                
# Defining Data loader class for feeding the model                                                          #                               
# Transforming segmention task to instance segmentation task employing MASK RCNN                            #
# author : Hamza Rafique https://github.com/ham952                                                          #                                                                                  #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license: #                                                        
# 1. Pytorch/torchvision : https://github.com/pytorch/vision                                                #                                                                                                   #
#############################################################################################################

import os
import numpy as np
import torch
import torch.utils.data
import shapely
import numpy as np
from PIL import Image, ImageDraw 
import json 
from shapely import wkt
from torch.utils.data import Dataset
from tifffile import imread

class segmDataset(Dataset):
    """
    Create Dataset for segmentation valid for both pre- and post-disaster images
    """
    
    def __init__(self, root, temporal = 'pre', transforms=None):
        """
         load all masks available pre_masks folder, sorting them to
         ensure that they are aligned
         then load the corresponding images in labels
        """
        self.root = root
        self.transforms = transforms
        self.temporal = temporal
        if self.temporal == 'post':
            self.masks = list(sorted(os.listdir(os.path.join(root, "post_masks"))))
        else:
            self.masks = list(sorted(os.listdir(os.path.join(root, "pre_masks"))))
            
        self.imgs = [mask.replace('tiff','png') for mask in self.masks]
        self.labels = [mask.replace('tiff','json') for mask in self.masks]
        
    def process_img(self,img_array, polygon_pts, scale_pct=0):
        """
        Converts polygons x,y cordinataes into bbox format
                Args:
                    img_array (numpy array): numpy representation of image.
                    polygon_pts (array): corners of the building polygon.
                Returns:
                    xmin,ymin,xmax,ymax
        """
        height, width, _ = img_array.shape

        xcoords = polygon_pts[:, 0]
        ycoords = polygon_pts[:, 1]
        xmin, xmax = np.min(xcoords), np.max(xcoords)
        ymin, ymax = np.min(ycoords), np.max(ycoords)

        xdiff = xmax - xmin
        ydiff = ymax - ymin

        #Extend image by scale percentage
        xmin = max(int(xmin - (xdiff * scale_pct)), 0)
        xmax = min(int(xmax + (xdiff * scale_pct)), width)
        ymin = max(int(ymin - (ydiff * scale_pct)), 0)
        ymax = min(int(ymax + (ydiff * scale_pct)), height)

        bbox = [xmin,ymin,xmax,ymax]
        return bbox

    def process_img_poly(self,img_path, label_path):
        """
        Extract Bounding Box coordinates for images using json label
                Args:
                    img_path : image path (string)
                    label_path: corresponding image json file path (string)
                Returns:
                     
        """
        bbox = [] 
        img_obj = Image.open(img_path)

        img_array = np.array(img_obj)

        #Get corresponding label for the current image
        label_file = open(label_path)
        label_data = json.load(label_file)

        #Find all polygons in a given image
        for feat in label_data['features']['xy']:

            poly_uuid = feat['properties']['uid'] + ".png"

            # Extract the polygon from the points given
            polygon_geom = shapely.wkt.loads(feat['wkt'])
            polygon_pts = np.array(list(polygon_geom.exterior.coords))
            bbox1 = self.process_img(img_array, polygon_pts)

            bbox.append(bbox1)
            

        return bbox

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])

        if self.temporal == 'post':
            mask_path = os.path.join(self.root, "post_masks", self.masks[idx])
        else:
            mask_path = os.path.join(self.root, "pre_masks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = imread(mask_path)
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]


        bbox = self.process_img_poly(img_path,label_path)
        # there is only one class
        num_objs = len(bbox)

        # to check bbox qty equal to number of objects
        assert num_objs == len(obj_ids)
        if len(bbox)!= len(obj_ids):
            print(len(bbox),len(obj_ids))
            print (img_path)
        
        boxes = torch.as_tensor(bbox, dtype=torch.float32)
        if self.temporal == 'post':
            labels = torch.tensor(obj_ids, dtype=torch.int64)
        else:
            labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

