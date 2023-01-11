import os, random
import numpy as np
import torch



# experiment reproduction
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)



root = os.getcwd()
datapath = os.path.join(root, 'data')

paths = dict(

    root=root, data=datapath,  model=os.path.join(datapath, 'model'),

    liebherr=os.path.join(datapath, 'liebherr'),
    liebherrvideos=os.path.join(datapath, 'liebherr', 'liebherrvideos'), 
        liebherrallvideos=os.path.join(datapath, 'liebherr', 'liebherrvideos', 'liebherrallvideos'),
        liebherrtrainvideos=os.path.join(datapath, 'liebherr', 'liebherrvideos', 'liebherrtrainvideos'),
        liebherrtestvideos=os.path.join(datapath, 'liebherr', 'liebherrvideos', 'liebherrtestvideos'),
    liebherrdataset=os.path.join(datapath, 'liebherr', 'liebherrdataset'), 
        liebherrdatasettrain=os.path.join(datapath, 'liebherr', 'liebherrdataset', 'liebherrdatasettrain'), 
        liebherrdatasettest=os.path.join(datapath, 'liebherr', 'liebherrdataset', 'liebherrdatasettest'),
    liebherriframes=os.path.join(datapath, 'liebherr', 'liebherriframes'), 
        liebherrtrainiframes=os.path.join(datapath, 'liebherr', 'liebherriframes', 'liebherrtrainiframes'),
        liebherrtestiframes=os.path.join(datapath, 'liebherr', 'liebherriframes', 'liebherrtestiframes'),

    # liebhertraincsv = os.path.join(datapath, 'liebherr', 'liebherrdataset', 'train.csv'),
    # liebhertrestcsv = os.path.join(datapath, 'liebherr', 'liebherrdataset', 'test.csv'),
    liebherrcsv = os.path.join(datapath, 'liebherr', 'liebherrdataset'),
    
    vision=os.path.join(datapath, 'vision'),
    visionvideos=os.path.join(datapath, 'vision', 'visionvideos'), 
        visionallvideos=os.path.join(datapath, 'vision', 'visionvideos', 'visionallvideos'),
        visiontrainvideos=os.path.join(datapath, 'vision', 'visionvideos', 'visiontrainvideos'),
        visiontestvideos=os.path.join(datapath, 'vision', 'visionvideos', 'visiontestvideos'),
    visiondataset=os.path.join(datapath, 'vision', 'visiondataset'), 
        visiondatasettrain=os.path.join(datapath, 'vision', 'visiondataset', 'visiondatasettrain'), 
        visiondatasettest=os.path.join(datapath, 'vision', 'visiondataset', 'visiondatasettest'),
    visioniframes=os.path.join(datapath, 'vision', 'visioniframes'), 
        visiontrainiframes=os.path.join(datapath, 'vision', 'visioniframes', 'visiontrainiframes'),
        visiontestiframes=os.path.join(datapath, 'vision', 'visioniframes', 'visiontestiframes'),

    # visiontraincsv=os.path.join(datapath, 'vision', 'visiondataset', 'train.csv'),
    # visiontestcsv=os.path.join(datapath, 'vision', 'visiondataset', 'test.csv')
    visioncsv=os.path.join(datapath, 'vision', 'visiondataset'),



)


def creatdir(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(e)

def allpaths(dirs: dict):
    for ke, val in dirs.items():
        creatdir(val)

def main():
    print(root)
    allpaths(dirs=paths)




if __name__ == '__main__':
    main()