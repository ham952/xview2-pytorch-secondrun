from xbdutils import segmDataset
import torch
import utils
from tqdm import tqdm
import transforms as T

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
        dataset = segmDataset('train/',temporal, get_transform(train=False))
        dataset_test = segmDataset('hold/',temporal,get_transform(train=False))
    else:
        dataset = segmDataset(data_path,temporal)
        dataset_test = segmDataset(data_path,temporal)

    return dataset, dataset_test


def main(args):
    
    dataset,dataset_test = get_dataset(temporal ='pre')

    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=utils.collate_fn)
    
    val_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=utils.collate_fn)

    dataiter = iter(data_loader)

    for i in tqdm(range(len(dataset))):
        images, labels = dataiter.next()
    
    dataiter_val = iter(val_loader)

    for i in tqdm(range(len(dataset_test))):
        images, labels = dataiter_val.next()

    print ('Success','\ntotal files tested  : {} '.format(len(dataset)+len(dataset_test)))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser( description = __doc__ )
    parser.add_argument('--train-dir', default = 'train/' , help = 'path to dataset dir')

    args = parser.parse_args()
    main(args)
    
