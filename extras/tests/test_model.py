#############################################################################################################
# xView2 Challenge                                                                                          #                                                                
# Segmentation task                                                                                         #                               
# Transforming segmention task to instance segmentation task employing MASK RCNN                            #
# author : Hamza Rafique https://github.com/ham952                                                          #                                                                                  #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license: #                                                        
# 1. Pytorch/torchvision : https://github.com/pytorch/vision                                                #
# 2. COCO API : https://github.com/cocodataset/cocoapi                                                      #                                                                                                   #
#############################################################################################################
import numpy as np
#import torch
import torchvision
#from torchvision.models.detection import MaskRCNN
#from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



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

def get_instance_segmentation_model_mobilenet(num_classes):

    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280
     
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios 
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
     
    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                                  output_size=14,
                                                                 sampling_ratio=2)
     
    # put the pieces together inside a FasterRCNN model
    model = MaskRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler,
                       mask_roi_pool=mask_roi_pooler)

    return model

model.eval()
x = [np.randn(3, 300, 400), np.randn(3, 500, 400)]
#img_path = 
predictions = model(x)

print(predictions)
'''
