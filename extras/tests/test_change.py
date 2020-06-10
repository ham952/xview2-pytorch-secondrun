from version2.xbdutils import segmDataset
#import torch
from version2 import utils
from tqdm import tqdm
from version2 import transforms as T
import numpy
from PIL import Image, ImageDraw 

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))        
        transforms.append(T.Resize(1024))
        
    return T.Compose(transforms)

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

print(images[0].shape)


#print(images)
img1 = Image.fromarray(images[0].mul(255).permute(1, 2, 0).byte().numpy())
draw = ImageDraw.Draw(img1)
#
print(targets[0]['boxes'].shape)
print(targets[0]['boxes'][0])

for box in targets[0]['boxes']:
    box = box.numpy()
    draw.rectangle(box)
print(targets[0]['masks'].shape)
img1.show()
'''
print(images[0].shape)
print(targets[0]['masks'].shape)

mask = targets[0]['masks'][0]
mask = mask.numpy()
print(numpy.unique(mask))
'''

