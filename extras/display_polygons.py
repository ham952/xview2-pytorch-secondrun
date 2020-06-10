#############################################################################################################
# xView2 Challenge                                                                                          #                                                                
# Display polygons from labels and predictions                                                              #
# author : Hamza Rafique https://github.com/ham952                                                          #                                                                                  #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license: #                                                        
# 1. Pytorch/torchvision : https://github.com/pytorch/vision                                                #                                                                                                   #
#############################################################################################################

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import logging
import cv2
import os
import json 
from shapely import wkt

#ind = '00000241' # multiple small minor , major damage inst
ind = '00000889'
#ind = '00000235'

#ind = '00000031'
IMG_PATH_PRE = './test/images/socal-fire_'+ind+'_pre_disaster.png'
IMG_PATH_POST = './test/images/socal-fire_'+ind+'_post_disaster.png' 
LAB_PATH_PRE = './test/labels/socal-fire_'+ind+'_pre_disaster.json'
LAB_PATH_POST = './test/labels/socal-fire_'+ind+'_post_disaster.json' 
CLS_MODEL = './models/model_cls_35.pth'
#LOC_MODEL = './models/model_loc_tr_115.pth'
LOC_MODEL = './models/model_loc_160.pth'
SAVE_DIR = './results'
DISP_LABELS = False
SAVE_IMG = False
INSTANCE_CATEGORY_NAMES = ['__background__', 'no-damage','minor-damage','major-damage','destroyed']



def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def _get_prediction(img_path, model, threshold):
  """
  get_prediction
    parameters:
      - img_path - path of the input image
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
        ie: eg. segment of cat is made 1 and rest of the image is made 0
    
  """
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  pred = model([img])
  pred_score = list(pred[0]['scores'].detach().numpy())
  if len(pred_score) == 0 or np.max(pred_score)<threshold:  # to cater for 1. no prediction 2. less then threshold
    return None,None,None,None
  pred_t = [pred_score.index(x) for x in pred_score if x>=threshold][-1]
  masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
  pred_class = [INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
  labels = list(pred[0]['labels'].detach().numpy())
  masks = masks[:pred_t+1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  pred_labels = labels[:pred_t+1]
  
  return masks, pred_boxes, pred_class, pred_labels

def _colour_masks(image, pred_labels):

  damage_dict = {
    "no-damage": [0, 128, 0],
    "minor-damage": [0, 0, 255],
    "major-damage": [255, 0,255],
    "destroyed": [255, 0, 0]
}

  _damage_dict = {
    1: [0, 255, 0],
    2: [0, 0, 255],
    3: [255, 69, 0],
    4: [255, 0, 0]
}
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  #r[image == 1], g[image == 1], b[image == 1] = damage_dict[pred_class]
  r[image == 1], g[image == 1], b[image == 1] = _damage_dict[pred_labels]
  coloured_mask = np.stack([r, g, b], axis=2)

  return coloured_mask

def _instance_mask(img_path, model, threshold=0.5, rect_th=1, text_size=1, text_th=1):
  """
  instance_segmentation_api
    parameters:
      - img_path - path to input image
    method:
      - prediction is obtained by get_prediction
      - each mask is given random color
      - each mask is added to the image in the ration 1:0.8 with opencv
      - final output is displayed
  """
  masks, boxes, pred_cls, pred_lab = _get_prediction(img_path, model, threshold)
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  if masks is not None:
    for i in range(len(masks)):
      rgb_mask = _colour_masks(masks[i],pred_lab[i])
      img = cv2.addWeighted(img, 1, rgb_mask, 1, 0)
      cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
      if DISP_LABELS:
          cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
          
  return img

def _get_gt(img_path,label_path):
        
    path_to_label_value = label_path
    path_to_image_value = img_path

    with open(path_to_label_value, 'rb') as image_json_file:
        image_json = json.load(image_json_file)

    wkt_polygons = []
    coords = image_json['features']['xy']


    for coord in coords:
        if 'subtype' in coord['properties']:
            damage = coord['properties']['subtype']
        else:
            damage = 'no-damage'
        wkt_polygons.append((damage, coord['wkt']))
        
    polygons = []

    for damage, swkt in wkt_polygons:
        polygons.append((damage, wkt.loads(swkt)))

    img = Image.open(path_to_image_value)


    draw = ImageDraw.Draw(img, 'RGBA')
    
    damage_dict = {
        "no-damage": (0, 255, 0, 125),
        "minor-damage": (0, 0, 255, 125),
        "major-damage": (255, 0, 255, 125),
        "destroyed": (255, 0, 0, 125),
        "un-classified": (255, 255, 255, 125)
    }

    for damage, polygon in polygons:
        x,y = polygon.exterior.coords.xy
        coords = list(zip(x,y))
        draw.polygon(coords, damage_dict[damage])

    del draw

    return img

def _load_weights(num_class,model_weights):
    
    model = get_instance_segmentation_model(num_class)
    checkpoint = torch.load(model_weights, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    return model

def main():

    # load models and weights
    logging.info('Loading models and weights')
    model_cls = _load_weights(5,CLS_MODEL)
    model_loc = _load_weights(2,LOC_MODEL)

    #perform predictions and generate polygons
    logging.info('Performing predictions using loaded weights')
    img_poly_post = _instance_mask(IMG_PATH_POST, model_cls)
    img_poly_pre = _instance_mask(IMG_PATH_PRE, model_loc)

    #get ground truth for polygons
    logging.info('Fetching ground truth of images')
    gt_post = _get_gt(IMG_PATH_POST, LAB_PATH_POST)
    gt_pre = _get_gt(IMG_PATH_PRE, LAB_PATH_PRE)

    ## DISPLAT
    logging.info('Displaying results')
    my_dpi = 96
    fig_size = 10800
    fig_loc = plt.figure(figsize=(fig_size/my_dpi, fig_size/my_dpi), dpi=my_dpi)

    ax1 = fig_loc.add_subplot(1,2,1)
    ax1.set_title('GROUND TRUTH : PRE DISASTER IMAGE')
    ax1.imshow(gt_pre)
    ax2 = fig_loc.add_subplot(1,2,2)
    ax2.set_title('PREDICTIONS : PRE DISASTER IMAGE')
    ax2.imshow(img_poly_pre)
    
    fig_cls = plt.figure(figsize=(fig_size/my_dpi, fig_size/my_dpi), dpi=my_dpi)

    ax1 = fig_cls.add_subplot(1,2,1)
    ax1.set_title('GROUND TRUTH : POST DISASTER IMAGE')
    ax1.imshow(gt_post)
    ax2 = fig_cls.add_subplot(1,2,2)
    ax2.set_title('PREDICTIONS : POST DISASTER IMAGE')
    ax2.imshow(img_poly_post)

    ## Saving images
    if SAVE_IMG:
        logging.info('Saving Inferences')
        fig_loc.savefig(SAVE_DIR+'/'+ 'pre_disaster.png',dpi=my_dpi)
        fig_cls.savefig(SAVE_DIR+'/'+ 'post_disaster.png',dpi=my_dpi)

    plt.show()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()


