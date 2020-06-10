#############################################################################################################
# xView2 Challenge                                                                                          #                                                                
# Analysing the statistics of post disaster images                                                          #                               
# Analysing individual types and their count                                                                #
# author : Hamza Rafique https://github.com/ham952                                                          #                                                                                  #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license: #                                                        
# 1. Pytorch/torchvision : https://github.com/pytorch/vision                                                #                                                                                                   #
#############################################################################################################

import torch
import os
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np
import torch
import torch.utils.data
import shapely
from PIL import Image, ImageDraw 
import json 
from shapely import wkt
from torch.utils.data import Dataset
from tifffile import imread


class statsDataset(Dataset):
    """
    Create Dataset for segmentation using Only pre-disaster images
    """
    
    def __init__(self, root, transforms=None):
        """
         load all masks available pre_masks folder, sorting them to
         ensure that they are aligned
         then load the corresponding images in labels
        """
        self.root = root
        self.transforms = transforms
        
        self.masks = list(sorted(os.listdir(os.path.join(root, "post_masks"))))
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
        mask_path = os.path.join(self.root, "post_masks", self.masks[idx])

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


        bbox = self.process_img_poly(img_path,label_path)
        # there is only one class
        num_objs = len(bbox)
        labels = torch.tensor(obj_ids, dtype=torch.int64)

        # to check bbox qty equal to number of objects
        assert num_objs == len(obj_ids)
        if len(bbox)!= len(obj_ids):
            print(len(bbox),len(obj_ids))
            print (img_path)
        

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        #return img, target
        return img_path,num_objs, labels

    def __len__(self):
        return len(self.imgs)


def generate_csv(dataset_dir,csv_path="./",activity="Post"):

    if activity == "Pre":
        csv_name = "pre_disaster.csv"
    else:
        csv_name = "post_disaster.csv"
        
    dataset = statsDataset(dataset_dir)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)


    dataiter = iter(data_loader)
    x0_data = []
    x1_data = []
    x2_data = []
    y_data = []
    no_damage_count = []
    minor_damage_count = []
    major_damage_count = []
    destroyed_count = []
    
    for i in tqdm(range(len(dataset))):
    #for i in tqdm(range(10)):
        
        image, inst, labels = dataiter.next()
        no_damage = minor_damage = major_damage = destroyed = 0

        label, count = np.unique(labels,return_counts = True)

        for j in range(len(label)) :
            if label[j] == 1:
                no_damage = count[j]
            elif label[j] == 2:
                minor_damage = count[j]
            elif label[j] == 3:
                major_damage = count[j]
            else:
                destroyed = count[j]
        
        no_damage_count.append(no_damage)
        minor_damage_count.append(minor_damage)
        major_damage_count.append(major_damage)
        destroyed_count.append(destroyed)
        
        tmp = image[0].split("\\")[1]
        tmp1 = tmp.split("_")[0]
        
        if tmp1.split("-")[0] == 'hurricane':
            x0_data.append(tmp1.split("-")[0])
        elif tmp1.split("-")[0] == 'santa':
            x0_data.append(tmp1.split("-")[2])
        else:
            x0_data.append(tmp1.split("-")[1])
            
        x1_data.append(tmp.split("_")[0])
        x2_data.append(tmp.split("_")[1])
        y_data.append(inst.data.numpy()[0])

    output_train_csv_path = os.path.join(csv_path, csv_name )
    data_array = {'disaster_type': x0_data,'disaster_name': x1_data,'img_id': x2_data, 'instance_count': y_data,
                  'no-damage' : no_damage_count, 'minor-damage': minor_damage_count, 'major-damage' : major_damage_count, 'destroyed' : destroyed_count }
    df = pd.DataFrame(data = data_array)
    df.to_csv(output_train_csv_path)
    print ('Success','\ntotal files tested  : {} '.format(len(y_data)))

generate_csv('train/')


df = pd.read_csv('post_disaster.csv',index_col=0)
total = df['instance_count'].count()
total_inst = df['instance_count'].sum()

print('Total number of images : ',total)
print('Total number of instances : ',df['instance_count'].sum())

print('\n Overall Statistics of Instances\n\n',df['instance_count'].describe())

print('Total number of no-damage instances : {} ({:.2f})% '.format(df['no-damage'].sum(), ((df['no-damage'].sum()) / total_inst ) * 100 ))
print('Total number of minor-damage instances : {} ({:.2f})% '.format(df['minor-damage'].sum(), ((df['minor-damage'].sum()) / total_inst ) * 100 ))
print('Total number of major-damage instances : {} ({:.2f})% '.format(df['major-damage'].sum(), ((df['major-damage'].sum()) / total_inst ) * 100 ))
print('Total number of destroyed instances : {} ({:.2f})% '.format(df['destroyed'].sum() , ((df['destroyed'].sum()) / total_inst ) * 100 ))


print('Instances Distribution wrt types of Disasters :')


def gen_stats(cat='disaster_type'):
    
    disaster_type = []
    count = []
    inst_count = []
    no_damage = []
    minor_damage = []
    major_damage = []
    destroyed = []

    for dt in df[cat].unique():

        disaster_type.append(dt)
        count.append(df['instance_count'][df[cat]==dt].count())
        inst_count.append(df['instance_count'][df[cat]==dt].sum())

        tot = df['instance_count'][df[cat]==dt].sum()

        num = df['no-damage'][df[cat]==dt].sum()
        no_damage.append((num,format( (num/tot)*100 , '.2f')) )

        num1 = df['minor-damage'][df[cat]==dt].sum()
        minor_damage.append((num1,format( (num1/tot)*100, '.2f')) )

        num2 = df['major-damage'][df[cat]==dt].sum()
        major_damage.append((num2,format( (num2/tot)*100, '.2f')) )

        num3 = df['destroyed'][df[cat]==dt].sum()
        destroyed.append((num3,format( (num3/tot)*100 , '.2f')) )


    data = {
            'Disaster Type':disaster_type,
            'Image Count' : count,
            'Instance Count': inst_count,
            'no-damage, %age' : no_damage,
            'minor-damage, %age' : minor_damage,
            'major-damage, %age' : major_damage,
            'destroyed, %age' : destroyed
            }
        
    disaster_types = pd.DataFrame(data)

    return disaster_types

path1 = os.path.join("./", 'disaster_types.csv' )
disaster_types = gen_stats(cat='disaster_type')
disaster_types.to_csv(path1)
print(disaster_types)

path2 = os.path.join("./", 'disaster_names.csv' )
disaster_names = gen_stats(cat='disaster_name')
disaster_names.to_csv(path2)
print(disaster_names)

