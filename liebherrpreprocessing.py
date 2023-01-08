import os, random
import cv2
import conf as cfg
import numpy as np



# remove .DS_Stor
def ds_remove(array):
    try:
        array.remove('.DS_Store')
    except Exception as e:
        print('f{e}') 
    
    return array


# rename files
def renames(folderpath):
    files = os.listdir(folderpath)
    files = ds_remove(files)
    i=0
    for file in files:
        oldname = os.path.join(folderpath, file)
        newname = os.path.join(folderpath, f'video_{i}.avi')
        os.rename(oldname, newname)
        i+=1

def allrename(folderpath):
    listfolders = os.listdir(folderpath)
    listfolders = ds_remove(listfolders)
    for folder in listfolders:
        dirpath = os.path.join(folderpath, folder)
        renames(dirpath)

# iframe extraction command for iframe extraction by ffmpeg
def iframes(videopath, trgiframespath):
    videoname = videopath.split('/')[-1].strip()
    filepath = os.path.join(trgiframespath, f'{videoname}-image-')
    cfg.creatdir(trgiframespath)
    command = f"ffmpeg -skip_frame nokey -i {videopath} -vsync 0 -frame_pts true {filepath}out%d.png"
    os.system(command)


def iframe_extraction(videosfolderpath, trgiframefolderpath):
    # [Gantry Travel 1, ...]
    srcvideofolders = os.listdir(videosfolderpath)
    srcvideofolders = ds_remove(srcvideofolders)

    for srcvideofolder in srcvideofolders:
        # [2022_9_..., ....]
        srcvideofilepath = os.path.join(videosfolderpath, srcvideofolder)
        srcvideofiles = os.listdir(srcvideofilepath)
        srcvideofiles = ds_remove(srcvideofiles)
        random.shuffle(srcvideofiles)
        trainfiles = srcvideofiles[1:]
        testfiles = srcvideofiles[0]

        # for i in range(10):
        #     print(trainfiles[i])

        for trainfile in trainfiles:
            filepath = os.path.join(videosfolderpath, srcvideofilepath, trainfile)
            trgfilepath = os.path.join(trgiframefolderpath, 'liebherrtrainiframes', srcvideofolder)
            iframes(videopath=filepath, trgiframespath=trgfilepath)

        # for testfile in testfiles:
        filepath = os.path.join(videosfolderpath, srcvideofilepath, testfiles)
        trgfilepath = os.path.join(trgiframefolderpath, 'liebherrtestiframes', srcvideofolder)
        iframes(videopath=filepath, trgiframespath=trgfilepath)





def main():
    path = os.path.join('hello', 'world')
    # iframes(videopath=path, trgiframespath='jk')

    srcfoldervideos = cfg.paths['liebherrallvideos']
    trgiframepath = cfg.paths['liebherriframes']
    iframe_extraction(videosfolderpath=srcfoldervideos, trgiframefolderpath=trgiframepath)


if __name__ == '__main__':
    main()