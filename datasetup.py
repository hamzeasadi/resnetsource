import torch
import os
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import conf as cfg
import imageio
import cv2
import pandas as pd





dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# remove .DS_Stor
def ds_remove(array):
    try:
        array.remove('.DS_Store')
    except Exception as e:
        print('f{e}') 
    
    return array


class SoureData(Dataset):
    """
    doc
    """
    def __init__(self, datapath: str, train=False):
        super().__init__()

        if train:
            self.path = os.path.join(cfg.paths[datapath], 'train.csv')
        else:
            self.path = os.path.join(cfg.paths[datapath], 'test.csv')

        self.fileids = pd.read_csv(self.path, header=None).values
        self.t = transforms.Compose([transforms.ToTensor(), transforms.Grayscale(),  transforms.Normalize(mean=[126/255], std=[200/255])])

    def __len__(self):
        return len(self.fileids)

    def __getitem__(self, idx):
        patchinfo = self.fileids[idx]
        imgpath = patchinfo[0][2:-1]
        img = cv2.imread(imgpath)
        imgt = self.t(img)
        coord = self.coord(info=patchinfo)
        sample = torch.cat((imgt, coord), dim=0)

        cls = int(patchinfo[-1][:-1])

        return sample, torch.tensor(cls)

    def coord(self, info):
        """
        info format: (imgpath, h, w, hi, wi)
        """
        h, w, hi, wi = info[1], info[2], info[3], info[4]

        channelx = torch.ones(size=(224, 224))
        for i in range(hi, hi+224):
            channelx[i-hi, :] = i*channelx[i-hi, :]
        channelx = 2*(channelx/h) - 1

        channely = torch.ones(size=(224, 224))
        for i in range(wi, wi+224):
            channely[:, i-wi] = i*channely[:, i-wi]
        channely = 2*(channely/w) - 1
        
        return torch.cat((channelx.unsqueeze(dim=0), channely.unsqueeze(dim=0)), dim=0)




def createdl(dataset, train_percent, batch_size):
    l = len(dataset)
    train_size = int(l*train_percent)
    valid_size = l - train_size
    train_data, valid_data = random_split(dataset=dataset, lengths=[train_size, valid_size])

    trainl, validl = DataLoader(train_data, batch_size=batch_size, shuffle=True), DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    return trainl, validl


# vision
vtrainvalid = SoureData(datapath='visioncsv', train=True)
vtest = SoureData(datapath='visioncsv', train=False)

vtrainl, vvalidl = createdl(dataset=vtrainvalid, batch_size=128, train_percent=0.85)
vtestl = DataLoader(vtest, batch_size=128)

# liebherr
ltrainvalid = SoureData(datapath='liebherrcsv', train=True)
ltest = SoureData(datapath='liebherrcsv', train=False)

ltrainl, lvalidl = createdl(dataset=ltrainvalid, batch_size=128, train_percent=0.85)
ltestl = DataLoader(ltest, batch_size=128)

datasets = dict(
    vision=(vtrainl, vvalidl, vtestl), liebherr=(ltrainl, lvalidl, ltestl)
)

def main():
    datasetpath = 'liebherrcsv'
    # dataset = SoureData(datapath=datasetpath, train=False)
    # ltestl = DataLoader(dataset, batch_size=128)
    # for batch in ltestl:
    #     X = batch[0]
    #     x = X[0]
    #     for i in range(3):
    #         cv2.imshow('img', x[i].numpy())
    #         cv2.waitKey(0)

    print(vtrainl)




if __name__ =='__main__':
    main()




