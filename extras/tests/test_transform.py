from version2.xbdutils import segmDataset
import torch
from version2 import utils
from version2 import transforms as T
import numpy
from PIL import Image, ImageDraw 

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    transforms.append(T.Resize(800))
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomRotateCW(0.5))
        transforms.append(T.RandomRotateCCW(0.5))
        transforms.append(T.RandomHue(0.2,hue = [-0.5,0.5]))
        transforms.append(T.RandomBrightness(0.3 ,brightness = [0.4,0.8]))
        transforms.append(T.RandomContrast(0.7 ,contrast = [0.3,0.6])) # good for cloud cover emulation
        transforms.append(T.RandomSaturation(0.2 ,saturation = [0,1]))
        
    return T.RandomOrderCustom(transforms)

def get_dataset(data_path = 'train/', use_transform = True):

 
    if use_transform:
        dataset = segmDataset('train/',get_transform(train=True))
        dataset_test = segmDataset('train/',get_transform(train=False))
    else:
        dataset = segmDataset(data_path)
        dataset_test = segmDataset(data_path)

    return dataset, dataset_test


dataset,dataset_test = get_dataset()

data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)


dataiter = iter(data_loader)


images, targets = dataiter.next()

print('Shape of images : ',images[0].shape)


#print(images)
#im = torch.rot90(images[0],1,[1,2])
#img1 = Image.fromarray(im.mul(255).permute(1, 2, 0).byte().numpy())
img1 = Image.fromarray(images[0].mul(255).permute(1, 2, 0).byte().numpy())


draw = ImageDraw.Draw(img1)

print('Shape of bbox : ',targets[0]['boxes'].shape)
print('bbox example :',targets[0]['boxes'][0])




for box in targets[0]['boxes']:
    box = box.numpy()
    draw.rectangle(box)
print('Shape of masks : ',targets[0]['masks'].shape)
img1.show()

'''
print(images[0].shape)
print(targets[0]['masks'].shape)

mask = targets[0]['masks'][0]
mask = mask.numpy()
print(numpy.unique(mask))
'''
