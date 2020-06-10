# -*- coding: utf-8 -*-
#############################################################################################################
# xView2 Challenge                                                                                          #                                                                
# Segmentation task                                                                                         #                               
# Transforming segmention task to instance segmentation task employing MASK RCNN                            #
# author : Hamza Rafique https://github.com/ham952                                                          #                                                                                  #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license: #                                                        
# 1. Pytorch/torchvision : https://github.com/pytorch/vision                                                #
# 2. MIoU : https://github.com/CYBORG-NIT-ROURKELA/Improving_Semantic_segmentation                          #
# Changelog :                                                                                               #
# 1. Addition of MIoU metric for validation                                                                 #
# 2. Use of "hold" dataset for validation and "train" for training purposes                                 #
# 3. Removal of redundant transformations                                                                   #
#############################################################################################################
import numpy as np
import time
import datetime
from xbdutils import segmDataset
import torch
import math
import os
from os import chdir, getcwd
import logging
import transforms as T
import utils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    transforms.append(T.Resize(800))
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)

def get_dataset(temporal, data_path = 'train/', use_transform = True):

 
    if use_transform:
        dataset = segmDataset('train/',temporal, get_transform(train=True))
        dataset_test = segmDataset('hold/',temporal,get_transform(train=False))
    else:
        dataset = segmDataset(data_path,temporal)
        dataset_test = segmDataset(data_path,temporal)

    return dataset, dataset_test

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

def compute_miou(actual, pred, num_class=2):
    a = actual
    a = a.reshape((800*800,))
    a_count = np.bincount(a, weights = None, minlength = num_class) # A
    
    b = pred
    b = b.reshape((800*800,))
    b_count = np.bincount(b, weights = None, minlength = num_class) # B
    
    c = a * num_class + b
    cm = np.bincount(c, weights = None, minlength = num_class * num_class)
    cm = cm.reshape((num_class, num_class))
    
    Nr = np.diag(cm) # A ⋂ B
    Dr = a_count + b_count - Nr # A ⋃ B
    individual_iou = Nr/Dr
    miou = np.nanmean(individual_iou)
    
    return miou

@torch.no_grad()
def validate(epoch,model, data_loader, device, threshold = 0.50):

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    tmp = np.ones((800,800),dtype=np.uint8)
    
    for image, targets in metric_logger.log_every(epoch,data_loader, 25, header):
        image = list(img.to(device) for img in image)
        ## ToDo : Require size assertion

        
        targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]
        mask = targets[0]['masks'].numpy()
        target_mask = np.where(np.any(mask == 1, axis=0),tmp,0)
        

        #torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        evaluator_time = time.time()
        pred_score = list(outputs[0]['scores'].numpy())

        if len(pred_score) != 0 and np.max(pred_score)>threshold:  # to cater for 1. no prediction 2. less then threshold

            pred_t = [pred_score.index(x) for x in pred_score if x>=threshold][-1]
            masks = (outputs[0]['masks']>0.5).squeeze().numpy()
            masks = masks[:pred_t+1]
            pred_mask = np.where(np.any(masks == True, axis=0),tmp,0)

        else:
            pred_mask = np.zeros((800,800),dtype=np.uint8)    
        evaluator_time = time.time() - evaluator_time
        
        miou = compute_miou(target_mask, pred_mask)
        # cleaning memory
        del outputs, targets, target_mask, pred_mask
        
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time, MIoU = miou)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    torch.set_num_threads(n_threads)

    return metric_logger


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(epoch,data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def main(args):
    temporal = 'pre' # LOCALIZATION   
    logging.info(args)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device == 'cpu':
        num_workers = 0
    else:
        num_workers = 0
        
    # loading data
    logging.info("Loading data")    

    dataset, dataset_val = get_dataset(temporal)
    

    # define training and validation data loaders
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    indices_val = torch.randperm(len(dataset_val)).tolist()
    
    # Using all data for training
    dataset = torch.utils.data.Subset(dataset, indices[:])
    dataset_val = torch.utils.data.Subset(dataset_val, indices_val[:])

    # using 20 % out of 408 fire images for test
    #dataset = torch.utils.data.Subset(dataset, indices[:-85])
    #dataset_val = torch.utils.data.Subset(dataset_val, indices_val[-85:])

    logging.info("Traing images : {} , Test images : {}".format(len(dataset),len(dataset_val)))

    logging.info("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
            collate_fn=utils.collate_fn)

    val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=utils.collate_fn)

    # Creating model
    logging.info("Creating model")
    
    # our dataset has two classes only - background and building
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,momentum=0.9, weight_decay=0.0001)
    #optimizer = torch.optim.AdamW(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every lr_step epochs
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_steps,
                                                        gamma=args.lr_gamma)

    if args.resume:
        logging.info("Loading saved model for training")
        path = os.path.join(args.output_dir, 'model_{}.pth'.format(args.resume))
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #optimizer.param_groups[0]["lr"] = 0.0001
        #optimizer.param_groups[0]["weight_decay"] = 0.0005
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

        del checkpoint
        
    if args.test_only:
        logging.info("Using model only for evaluations")
        epoch = args.start_epoch 
        validate(epoch,model, val_loader, device=device)
        return
        
    num_epochs = args.epochs

    logging.info(f'''Starting training:
        Epochs:           {args.epochs}
        Starting Epoch:   {args.start_epoch}
        Batch Size:       {args.batch_size}
        Learning Rate:    {args.lr}
        Lr Scheduler :    {args.lr_steps}
        Lr Reduction:     {args.lr_gamma}
        Save Freq(epochs):{args.save_freq}
        Device:           {device}
    ''')

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        #train for one epoch, printing every 25 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=25)
        # update the learning rate
        lr_scheduler.step()
    
        if epoch >0 and epoch % args.save_freq == 0:
            logging.info('Saving model at epoch : {}'.format(epoch))            
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            
        # evaluate on the hold dataset
        validate(epoch,model, val_loader, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
    description=__doc__)
    
    handlers = [logging.FileHandler('results/log.log',mode='a'), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s',handlers = handlers)
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='size of images per batch')
    parser.add_argument('--epochs', default=251, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr-step-size', default=10, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[201], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--output-dir', default='./results', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpointi-e epoch')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--save-freq', default=20, type=int, help='model saving frequency')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
