from Localization.xbdutils import segmDataset
from Localization import transforms as T
import torch
from version2 import utils
from tqdm import tqdm
from PIL import Image
import numpy as np

# Read image masks
# Save image masks as being saved by test_loc
# compare saved mask with target mask
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())

    return T.Compose(transforms)

def get_dataset(temporal = 'pre', data_path = 'train/', use_transform = True):

 
    if use_transform:
        dataset = segmDataset('train/',temporal, get_transform(train=True))
        dataset_test = segmDataset('train/',temporal,get_transform(train=False))
    else:
        dataset = segmDataset(data_path,temporal)
        dataset_test = segmDataset(data_path,temporal)

    return dataset, dataset_test


def main(args):
    temporal = 'pre'
    dataset,dataset_test = get_dataset(data_path = args.train_dir)

    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=utils.collate_fn)


    dataiter = iter(data_loader)

    #for i in tqdm(range(len(dataset))):
    for i in range(2):
        images, labels = dataiter.next()
        
    print(labels[0]['masks'].shape)
    print(images[0].shape)

    masks = labels[0]['masks'].numpy()
    print(np.shape(masks))

    tmp = np.ones((1024,1024),dtype=np.uint8)
    #mask = np.where(np.any(masks == True, axis=0),tmp,0)
    mask = np.where(np.any(masks == 1, axis=0),tmp,0)

    print(mask.shape)
    print(np.unique(mask))

    target = Image.open('guatemala-volcano_00000000_pre_disaster_target.png')
    target_mask = np.array(target)

    print(np.shape(target_mask))

    print((target_mask == mask).all())
    #img1 = Image.fromarray(images[0].mul(255).permute(1, 2, 0).byte().numpy())
    #img1.show()
    


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser( description = __doc__ )
    parser.add_argument('--train-dir', default = 'train/' , help = 'path to dataset dir')

    args = parser.parse_args()
    main(args)
    
