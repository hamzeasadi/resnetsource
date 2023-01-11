import os, random
import cv2
import conf as cfg
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt
import torch
import imageio
from torchvision.datasets import ImageFolder



# remove .DS_Stor
def ds_remove(array):
    try:
        array.remove('.DS_Store')
    except Exception as e:
        print('f{e}') 
    
    return array

# iframe extraction command for iframe extraction by ffmpeg
def iframes(videopath, trgiframespath):
    videoname = videopath.split('/')[-1].strip()
    filepath = os.path.join(trgiframespath, f'{videoname}-image-')
    cfg.creatdir(trgiframespath)
    command = f"ffmpeg -skip_frame nokey -i {videopath} -vsync 0 -frame_pts true {filepath}out%d.png"
    os.system(command)


def iframe_extraction(videosfolderpath, trgiframefolderpath):
    # [D03_Huawei..., ...]
    srcvideofolders = os.listdir(videosfolderpath)
    srcvideofolders = ds_remove(srcvideofolders)

    for srcvideofolder in srcvideofolders:
        # visionallvideos/D03_Huawei../videos
        srcvideofolderpath = os.path.join(videosfolderpath, srcvideofolder, 'videos')
        # [outdoor, outdoorYT, outdoorWA]
        outdoorfolders = os.listdir(srcvideofolderpath)
        outdoorfolders = ds_remove(outdoorfolders)
        for outdoorfolder in outdoorfolders:
            # visionallvideos/D03_Huawei../videos/outdoor
            outdoorfolderpath = os.path.join(srcvideofolderpath, outdoorfolder)
            # [DO3_V_outdorr..., ...]
            outdoorfoldervideos = os.listdir(outdoorfolderpath)
            outdoorfoldervideos = ds_remove(outdoorfoldervideos)
            random.shuffle(outdoorfoldervideos)
            trainfiles = outdoorfoldervideos[1:]
            testfiles = outdoorfoldervideos[0]

            for trainfile in trainfiles:
                filepath = os.path.join(outdoorfolderpath, trainfile)
                trgfilepath = os.path.join(trgiframefolderpath, 'visiontrainiframes', srcvideofolder)
                iframes(videopath=filepath, trgiframespath=trgfilepath)

            # for testfile in testfiles:
            filepath = os.path.join(outdoorfolderpath, testfiles)
            trgfilepath = os.path.join(trgiframefolderpath, 'visiontestiframes', srcvideofolder)
            iframes(videopath=filepath, trgiframespath=trgfilepath)


def addtocsv(csvpath, fileid):
    with open(csvpath, 'a') as f:
        f.write(f"{fileid}\n")
    

def imagepath(imgpath, H, W, trainpatc=False):
    img = cv2.imread(imgpath)
    h,w,c = img.shape

    if trainpatc:
        numh = 2*int(h/H) -1
        numw = 2*int(w/W) - 1
        dh, dw = int(H/2), int(W/2)

    else:
        numh = int(h/H)
        numw = int(w/W) 
        dh, dw = H, W

    patches = []
    for i in range(numh):
        hi = i*dh
        for j in range(numw):
            wi = j*dw
            patch = img[hi:hi+H, wi:wi+W, :]
            patches.append((patch, h, w, hi, wi))

    return patches



def allpatches(srcpath, trgpath, csvpath, H, W, trainpatch=False):
    srcfolders = os.listdir(srcpath)
    srcfolders = ds_remove(srcfolders)
    for i, srcfolder in enumerate(srcfolders):
        srcfolderpath = os.path.join(srcpath, srcfolder)
        srcfolderfiles = os.listdir(srcfolderpath)
        srcfolderfiles = ds_remove(srcfolderfiles)
        trgfolderpath = os.path.join(trgpath, srcfolder)
        cfg.creatdir(path=trgfolderpath)

        for j, srcfile in enumerate(srcfolderfiles):
            srcfilepath = os.path.join(srcfolderpath, srcfile)
            patches = imagepath(imgpath=srcfilepath, H=H, W=W, trainpatc=trainpatch)
            for k, patch in enumerate(patches):
                filename = f'folder_{i}_img-{j}_patch_{k}.png'
                filepath = os.path.join(trgfolderpath, filename)
                cv2.imwrite(filename=filepath, img=patch[0])

                if trainpatch:
                    csvfile = os.path.join(csvpath, 'train.csv')
                else:
                    csvfile = os.path.join(csvpath, 'test.csv') 

                fileid = (filepath, *patch[1:], i)
                addtocsv(csvpath=csvfile, fileid=fileid)
            # break
        # break




def main():
    path = '/Users/hamzeasadi/python/resnetsource/data/liebherr/liebherriframes/liebherrtestiframes/GantryTravel/video_16.avi-image-out0.png'

    # liebherr train
    srcpath = cfg.paths['liebherrtrainiframes']
    trgpath = cfg.paths['liebherrdatasettrain']
    csvpath = cfg.paths['liebherrcsv']
    allpatches(srcpath=srcpath, trgpath=trgpath, csvpath=csvpath, H=224, W=224, trainpatch=True)

    # liebherr test
    srcpath = cfg.paths['liebherrtestiframes']
    trgpath = cfg.paths['liebherrdatasettest']
    csvpath = cfg.paths['liebherrcsv']
    allpatches(srcpath=srcpath, trgpath=trgpath, csvpath=csvpath, H=224, W=224, trainpatch=False)

    # # vision train
    # srcpath = cfg.paths['visiontrainiframes']
    # trgpath = cfg.paths['visiondatasettrain']
    # csvpath = cfg.paths['visioncsv']
    # allpatches(srcpath=srcpath, trgpath=trgpath, csvpath=csvpath, H=224, W=224, trainpatch=True)

    # # vision test
    # srcpath = cfg.paths['visiontestiframes']
    # trgpath = cfg.paths['visiondatasettest']
    # csvpath = cfg.paths['visioncsv']
    # allpatches(srcpath=srcpath, trgpath=trgpath, csvpath=csvpath, H=224, W=224, trainpatch=False)




if __name__ == '__main__':
    main()
