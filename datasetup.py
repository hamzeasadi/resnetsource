import torch
import os
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import conf as cfg
import imageio
import cv2


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
    def __init__(self, datapath: str):
        super().__init__()
        self.path = cfg.paths[datapath]
        self.t = transforms.Compose([transforms.ToTensor()])
        self.files = self.file()

    def file(self):
        folders = os.listdir(self.path)
        folders = ds_remove(folders)
        listfiles = []
        for i, folder in enumerate(folders):
            files = os.listdir(os.path.join(self.path, folder))
            for f in files:
                filepath = os.path.join(self.path, folder, f)
                listfiles.append((i, filepath))

        return listfiles

    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cls, imgpath = self.files[idx]
        img = imageio.imread(imgpath)
        imgt = self.t(img)
        return imgt, torch.tensor(cls)



def create_loader(dataset: Dataset, batch_size: int=128, train_percent=0.85):
    l = len(dataset)
    train_size = int(l*train_percent)
    valid_size = l - train_size
    train_data, valid_data = random_split(dataset, lengths=[train_size, valid_size])

    return DataLoader(train_data, batch_size=batch_size, shuffle=True), DataLoader(valid_data, batch_size=batch_size, shuffle=True)




def main():
    datasetpath = 'liebherrdatasettest'
    dataset = SoureData(datapath=datasetpath)
    train_loader, valid_laoder = create_loader(dataset=dataset)
    for X, Y in valid_laoder:
        print(X.shape)
        print(Y)




if __name__ =='__main__':
    main()




