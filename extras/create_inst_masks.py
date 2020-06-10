#############################################################################################################
# xView2 Challenge                                                                                          #                                                                
# Creating instance masks for pre- and post-disaster images                                                 #                               
# Transforming segmention task to instance segmentation task employing MASK RCNN                            #
# author : Hamza Rafique https://github.com/ham952                                                          #                                                                                  #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license: #                                                        
# 1. Pytorch/torchvision : https://github.com/pytorch/vision                                                #                                                                                                   #
#############################################################################################################

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
from tifffile import imsave
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import cv2
import timeit
from os import path, makedirs, listdir
import sys
sys.setrecursionlimit(10000)
from multiprocessing import Pool
from tqdm import tqdm
from shapely.wkt import loads
from shapely.geometry import mapping, Polygon
import json

train_dirs = ['train','hold']
temporal = 'post' # 'pre','post'
masks_dir = temporal + '_masks'
LIMT_INST = False
CNT_INST = 255

def mask_for_polygon(poly, im_size=(1024, 1024)):
    img_mask = np.zeros(im_size, np.uint8)
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords)]
    interiors = [int_coords(pi.coords) for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


damage_dict = {
    "no-damage": 1.0,
    "minor-damage": 2.0,
    "major-damage": 3.0,
    "destroyed": 4.0,
    "un-classified": 1.0 # ?
}

def process_image_pre(json_file):
    js1 = json.load(open(json_file))

    msk = np.zeros((1024, 1024), dtype='float32')
    p = 254.0000
    i = 0.0001
    count = 0
    for feat in js1['features']['xy']:
        poly = loads(feat['wkt'])
        _msk = mask_for_polygon(poly)
        msk[_msk > 0] = p
        p+=i
        count +=1
        
    if LIMT_INST:
        if count == 0 or count >= CNT_INST:
            return json_file
        else:
            imsave(json_file.replace('labels', masks_dir).replace('_pre_disaster.json', '_pre_disaster.tiff'), msk,compress=6)
    else:
        if count == 0:
            return json_file
        else:
            imsave(json_file.replace('labels', masks_dir).replace('_pre_disaster.json', '_pre_disaster.tiff'), msk,compress=6)

def process_image_post(json_file):

    js2 = json.load(open(json_file.replace('_pre_disaster', '_post_disaster')))

    msk_damage = np.zeros((1024, 1024), dtype='float32')
    i = 0.001
    count = 0
    for feat in js2['features']['xy']:
        poly = loads(feat['wkt'])
        subtype = feat['properties']['subtype']
        _msk = mask_for_polygon(poly)
        msk_damage[_msk > 0] = damage_dict[subtype] + i
        i+= 0.001
        count +=1
        
    if LIMT_INST:
        if count == 0 or count >= CNT_INST:
            return json_file
        else:
            imsave(json_file.replace('labels', masks_dir).replace('_pre_disaster.json', '_post_disaster.tiff'), msk_damage,compress=6)
    else:
        if count == 0:
            return json_file
        else:
            imsave(json_file.replace('labels', masks_dir).replace('_pre_disaster.json', '_post_disaster.tiff'), msk_damage,compress=6)
            

if __name__ == '__main__':
    t0 = timeit.default_timer()

    all_files = []
    for d in train_dirs:
        makedirs(path.join(d, masks_dir), exist_ok=True)
        for f in sorted(listdir(path.join(d, 'images'))):
            if '_pre_disaster.png' in f:
                all_files.append(path.join(d, 'labels', f.replace('_pre_disaster.png', '_pre_disaster.json')))

    empty = []
    with Pool() as pool:
        if temporal == 'post':
            print('Creating masks for Post Disaster images')
            files = pool.map(process_image_post, all_files)
        else:
            print('Creating masks for Pre Disaster images')
            files = pool.map(process_image_pre, all_files)            
        empty.append(files)            

    empty = list(filter(None.__ne__, empty[0]))
    elapsed = timeit.default_timer() - t0

    print('Total images without any objects : ',len(empty))
    print('Total of {} masks generated from {} disaster images'.format(len(all_files)-len(empty), len(all_files)))
    print('Time: {:.3f} min'.format(elapsed / 60))
