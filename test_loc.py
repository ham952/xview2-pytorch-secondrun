#############################################################################################################
# xView2 Challenge                                                                                          #                                                                
# Inference task : Perform prediction using weights and save predicted masks                                #                               
# Transforming segmention task to instance segmentation task employing MASK RCNN                            #
# author : Hamza Rafique https://github.com/ham952                                                          #                                                                                  #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license: #                                                        
# 1. Pytorch/torchvision : https://github.com/pytorch/vision                                                #
# 2. COCO API : https://github.com/cocodataset/cocoapi                                                      #                                                                                                   #
#############################################################################################################

from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import torch
from tqdm import tqdm
import logging
import time
import datetime


def get_prediction(img_path, model, threshold, device):
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
  #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  img = Image.open(img_path)
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  image = img.to(device)
  
  t0 = time.time()
  pred = model([image])
  infer_time = time.time() - t0

  pred_score = list(pred[0]['scores'].detach().cpu().numpy())
  if len(pred_score) == 0 or np.max(pred_score)<threshold:  # to cater for 1. no prediction 2. less then threshold
    return None,None,None,infer_time
  
  pred_t = [pred_score.index(x) for x in pred_score if x>=threshold][-1]
  masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
  pred_class = [LOCALIZATION_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].detach().cpu().numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
  masks = masks[:pred_t+1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return masks, pred_boxes, pred_class, infer_time

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

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

def main(args):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # loading data
    logging.info("Loading data")
    test_images = os.listdir(args.input_dir)
    logging.info('Successfully loaded {} test images'.format(len(test_images)))
    
    # Creating model
    logging.info("Creating model")
    
    # our dataset has two classes only - background and building
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)
    
    logging.info("Loading model weights")
    path = os.path.join(args.weights_dir, 'model_{}.pth'.format(args.checkpoint))
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    logging.info('Performing Inference and generating prediction masks')
    start_time = time.time()

    tmp = np.ones((1024,1024),dtype=np.uint8)
    count = 0
    avg_time = []
    
    for i in tqdm(range(len(test_images))):
    #for i in tqdm(range(4)):

        test_image = test_images[i]

        # using only pre disaster : LOCALIZATION
        if 'pre' in test_image:

            img_path = os.path.join(args.input_dir,test_image)
            masks, boxes, pred_cls, infer_time = get_prediction(img_path,model, args.threshold,device)
            avg_time.append(infer_time)
          
            if masks is None :
              mask = np.zeros((1024,1024),dtype=np.uint8)
            else:
              mask = np.where(np.any(masks == True, axis=0),tmp,0)

            img = Image.fromarray(mask,'L')

            tmp1 = test_image.split('.')
            img_name_pre = tmp1[0]+'_prediction.'+tmp1[1]
            img_name_post = img_name_pre.replace('_pre_','_post_')  # Todo : edit this when classification model trained

            img.save(os.path.join(args.output_dir,img_name_pre))
            img.save(os.path.join(args.output_dir,img_name_post))  # Todo : edit this when classification model trained

            count+=1      

            
    avg = sum(avg_time)/len(avg_time)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total Test time : {}'.format(total_time_str))
    print('total predictions performed : {} '.format(count))
    print('Average Inference time per image : {} sec'.format(avg))

    

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(
    description=__doc__)
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser.add_argument('--threshold', default=0.50, type=float,
                        help='Threshold for detection')    
    parser.add_argument('--checkpoint', default='',required = True, help='resume from checkpoint i-e weights id')
    parser.add_argument('--weights-dir', default='./results', help='path where to load weights from')
    parser.add_argument('--input-dir', default='./test/images', help='path where to test dataset')
    parser.add_argument('--output-dir', default='./test/predictions', help='path where to save Predictions')

    args = parser.parse_args()
    LOCALIZATION_CATEGORY_NAMES = ['__background__', 'building']
    
    main(args)
    
#python test_loc.py --checkpoint 100 --output-dir ./test/predictions_100
