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




trf = transforms.Compose(transforms=[transforms.ToTensor(), transforms.Grayscale(), 
transforms.Normalize(mean=[140/255], std=[200/255])])

def imagepatcheswithcoords(imgpath, H, W):
    img = cv2.imread(imgpath)
    tgrayimg = trf(img)
    grayimg = tgrayimg.squeeze().numpy()
    # print(grayimg.shape)
    h, w = grayimg.shape

    channelx = np.ones(shape=(h, w))
    for i in range(h):
        channelx[i, :] = i*channelx[i, :]
    channelx = 2*(channelx/h) - 1

    channely = np.ones(shape=(h, w))
    for i in range(w):
        channely[:, i] = i*channely[:, i]
    channely = 2*(channely/w) - 1
    
    img = torch.randn(size=(h, w, 3)).numpy()
    img[:, :, 0] = grayimg
    img[:, :, 1] = channelx
    img[:, :, 2] = channely

    # for i in range(3):
    #     cv2.imshow('gray', img[:, :, i])
    #     cv2.waitKey(0)

    numh = 2*int(h/H) - 1
    numw = 2*int(w/W) - 1
    dh = int(H/2)
    dw = int(W/2)
    crops = []
    for i in range(numh):
        hi = i*dh
        for j in range(numw):
            wj = j*dw
            crop = img[hi:hi+H, wj:wj+W, :]
            crops.append(crop)

    return crops



def allpatches(srcpath, trgpath):
    srciframesfolders = os.listdir(srcpath)
    srciframesfolders = ds_remove(srciframesfolders)
    for iframefolder in srciframesfolders:
        iframes = os.listdir(os.path.join(srcpath, iframefolder))
        iframes = ds_remove(iframes)
        for j, iframe in enumerate(iframes):
            iframepath = os.path.join(srcpath, iframefolder, iframe)
            patches = imagepatcheswithcoords(imgpath=iframepath, H=224, W=224)
            patchpath = os.path.join(trgpath, iframefolder)
            cfg.creatdir(patchpath)
            for i, patch in enumerate(patches):
                patchname = os.path.join(patchpath, f"img-{j}-patch-{i}.tiff")
                print(patch)
                # cv2.imwrite(filename=patchname, img=patch)
                imageio.imsave(patchname, patch)
                break
                # cv2.imshow(patch)
           
            break
        # break




def main():
    path = os.path.join('hello', 'world')

    srcpath = os.path.join(cfg.paths['liebherrtrainiframes'])
    trgpath = os.path.join(cfg.paths['liebherrdatasettrain'])

    srcpath = os.path.join(cfg.paths['liebherrtestiframes'])
    trgpath = os.path.join(cfg.paths['liebherrdatasettest'])


if __name__ == '__main__':
    main()
