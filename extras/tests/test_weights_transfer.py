import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import torch


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

def _transfer_pretrained_weights(model, pretrained_checkpoint):

    #lodaing only weights
    pretrained_weights = pretrained_checkpoint['model']
    # removing output heads of cls, bbox, mask
    new_dict = {k.replace('module.',''):v for k, v in pretrained_weights.items()
                if 'cls_score' not in k and 'bbox_pred' not in k and 'mask_fcn_logits' not in k}
    this_state = model.state_dict()
    this_state.update(new_dict)
    model.load_state_dict(this_state)

    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 5
model = get_instance_segmentation_model(num_classes)

print('#######################  MODEL SUMMARY ###################')
print('\n\n')
print(print(model))

print('\n\n')
print('#######################  MODEL PARAMETERS ###################')
print('\n\n')
for k in model.state_dict():
  print(k,'\t',model.state_dict()[k].shape)

print('\n\n')
print('Loading checkpoint')
checkpoint = torch.load('./model/model_160.pth', map_location='cpu')
print('#######################  CHECKPOINT PARAMETERS ###################')
print('\n\n')
for key, val in checkpoint['model'].items():
  print (key,'\t',val.shape)

print('\n\n')
print('Transfering pre-trrained weights')

model = _transfer_pretrained_weights(model, checkpoint)
print( 'Success')

print('#######################  TRANSFERED MODEL PARAMETERS ###################')
print('\n\n')
for k in model.state_dict():
  print(k,'\t',model.state_dict()[k].shape)

print( 'Success')
