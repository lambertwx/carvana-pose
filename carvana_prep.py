# -*- coding: utf-8 -*-
"""
Code to prep images and validate the folder structure for training/validation/test data.

$Id$
@author: lambert.wixson
"""
from os import listdir, path
from os.path import isfile, join
from pathlib import Path
import re
from skimage import color, filters, io, transform

#%%
def getBasenamesInDir(dir, suffix):
    """
    Return a set containing the basenames of the xyz_##.suffix in a folder
    """
    #files = [f for f in listdir(dir) if isfile(join(dir, f)) and Path(f).suffix == ("." + suffix)]
    files = [f for f in listdir(dir) if Path(dir, f).is_file() and Path(f).suffix == ("." + suffix)]
    print("{0} files in {1}".format(len(files), dir))
    pat = re.compile('(.+)_[0-9][0-9].{0}$'.format(suffix))
    prefixes = [pat.match(f).group(1) for f in files]
    bases = set(prefixes)
    return(bases)

#%%
def checkAllPosesExist(dir, setBases, suffix, 
                       listPoses = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']):
    numChecked = 0
    numMissing = 0
    for base in setBases:
        for pose in listPoses:
            name = '{0}_{1}.{2}'.format(base, pose, suffix)
            full = Path(dir, name)
            numChecked += 1
            if not full.exists():
                print("Missing {0}".format(name))
                numMissing += 1
    print("Checked {0} files, missing {1}".format(numChecked, numMissing))
    return numMissing
    
#%%
def checkdirs(base, suffix, traindir, valdir, testdir,
              listPoses = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']):
    pathBase = Path(base)
    
    pathTrain = pathBase.joinpath(traindir)
    setTrain = getBasenamesInDir(pathTrain, suffix)
    print("Unique names in train: {0}".format(len(setTrain)))
   
    pathVal = pathBase.joinpath(valdir)
    setVal   = getBasenamesInDir(pathVal, suffix)
    print("Unique names in validate: {0}".format(len(setVal)))
    
    pathTest = pathBase.joinpath(testdir)
    setTest  = getBasenamesInDir(pathTest, suffix)
    print("Unique names in test: {0}".format(len(setTest)))
    
    # Make sure the count of unique names is correct for each set
    assert len(setTrain) == 3943
    assert len(setVal) == 1315
    assert len(setTest) == 1314
    
    # Make sure the sets don't intersect
    sect = setTrain.intersection(setVal)
    assert len(sect) == 0
    
    sect = setTrain.intersection(setTest)
    assert len(sect) == 0
    
    sect = setVal.intersection(setTest)
    assert len(sect) == 0
    
    print("All sets are disjoint.")
    
    # Make sure each folder has all of the images
    numMissing = checkAllPosesExist(pathTrain, setTrain, suffix, listPoses )
    if numMissing > 0:
        print("Files missing from train dir {0}".format(pathTrain))
        return False
    
    numMissing = checkAllPosesExist(pathVal, setVal, suffix, listPoses )
    if numMissing > 0:
        print("Files missing from val dir {0}".format(pathVal))
        return False
    
    numMissing = checkAllPosesExist(pathTest, setTest, suffix, listPoses )
    if numMissing > 0:
        print("Files missing from test dir {0}".format(pathTest))
        return False
    
    return True

#%%
def makegmag(img):
    gray = color.rgb2gray(img) * 255
    pyr = tuple(transform.pyramid_gaussian(gray, max_layer=3, mode = 'reflect'))
    gmag = filters.sobel_h(pyr[3])
    return gmag
#%%
    
def write_reduced(seq, destpath, suffix = None, grayscale=False):
    for f in range(len(seq)):
        img = seq[f]
        (dirname, filename) = path.split(seq.files[f])
        if suffix:
            # Restrict ourselves only to certain files.
            pat = re.compile('.+_{0}.jpg$'.format(suffix))
            if not pat.match(filename):
                continue
        if grayscale:
            gray = color.rgb2gray(img) 
            pyr = tuple(transform.pyramid_gaussian(gray, max_layer=3, mode = 'reflect'))
        else:
            pyr = tuple(transform.pyramid_gaussian(img, max_layer=3, mode = 'reflect'))
        out = (pyr[3] * 255).astype('uint8')
        
        filebase = str(Path(filename).with_suffix('.png'))
        dest = destpath + "/" + filebase
        io.imsave(dest, out)

#%%

#%%
if False:
    f = newshow(seq[0])
    videofig(len(seq), redraw_fn, winname="raw", cmap='gray',
         proc_func=lambda f: seq[f])
    videofig(len(seq), redraw_fn, winname="gmag",
         proc_func=lambda f: makegmag(seq[f]))
    base_data = "c:/Users/lambert.wixson/datasets/kaggle_carvana/"
    train_rawdir = "c:/Users/lambert.wixson/datasets/kaggle_carvana/train"
    train_dir = "c:/Users/lambert.wixson/datasets/kaggle_carvana/train/small"

    seq = io.imread_collection(base_data + "train/full/*.jpg", conserve_memory=True, check_files=False)
    write_reduced(seq, "c:/Users/lambert.wixson/datasets/kaggle_carvana/train/small-color")
    write_reduced(seq, "c:/Users/lambert.wixson/datasets/kaggle_carvana/train/small-color", "05")
    write_reduced(seq, "c:/Users/lambert.wixson/datasets/kaggle_carvana/train/small-color", "13")
    
    seq = io.imread_collection(base_data + "validate/full/*.jpg", conserve_memory=True, check_files=False)
    write_reduced(seq, base_data + "validate/small-color")
    write_reduced(seq, base_data + "validate/small-color", "05")
    write_reduced(seq, base_data + "validate/small-color", "13")
    
    seq = io.imread_collection(base_data + "extratrain/full/*.jpg", conserve_memory=True)
    write_reduced(seq, base_data + "extratrain/small-color")
    
    seq = io.imread_collection(base_data + "test/full/*.jpg", conserve_memory=True, check_files=False)
    write_reduced(seq, base_data + "test/small-color")
    write_reduced(seq, base_data + "test/small-color", "05")
    write_reduced(seq, base_data + "test/small-color", "13")

#%% Now check that all the folders are organized properly.
    foo = getBasenamesInDir(Path(base_data).joinpath('train/full'), 'jpg')
    checkAllPosesExist(Path(base_data).joinpath('train/full'), foo, 'jpg')
    
    checkdirs(base_data, 'jpg', 'train/full', 'validate/full', 'test/full')
# =============================================================================
#     63088 files in c:\Users\lambert.wixson\datasets\kaggle_carvana\train\full
#     Unique names in train: 3943
#     21040 files in c:\Users\lambert.wixson\datasets\kaggle_carvana\validate\full
#     Unique names in validate: 1315
#     21024 files in c:\Users\lambert.wixson\datasets\kaggle_carvana\test\full
#     Unique names in test: 1314
#     All sets are disjoint.
#     Checked 63088 files, missing 0
#     Checked 21040 files, missing 0
#     Checked 21024 files, missing 0
#     Out[32]: True
# =============================================================================
    
    checkdirs(base_data, 'png', 'train/small-color', 'validate/small-color', 'test/small-color', ['01', '05', '09', '13'])
# =============================================================================
# 7886 files in c:\Users\lambert.wixson\datasets\kaggle_carvana\train\small-color
# Unique names in train: 3943
# 2630 files in c:\Users\lambert.wixson\datasets\kaggle_carvana\validate\small-color
# Unique names in validate: 1315
# 2628 files in c:\Users\lambert.wixson\datasets\kaggle_carvana\test\small-color
# Unique names in test: 1314
# All sets are disjoint.
# Checked 7886 files, missing 0
# Checked 2630 files, missing 0
# Checked 2628 files, missing 0
# =============================================================================
    checkdirs(base_data, 'png', 'project_front_vs_back/train-color/yaw01', 'project_front_vs_back/validate-color/yaw01', 
              'project_front_vs_back/test-color/yaw01', ['01'])
    checkdirs(base_data, 'png', 'project_front_vs_back/train-color/yaw05', 'project_front_vs_back/validate-color/yaw05', 
              'project_front_vs_back/test-color/yaw05', ['05'])
    checkdirs(base_data, 'png', 'project_front_vs_back/train-color/yaw09', 'project_front_vs_back/validate-color/yaw09', 
              'project_front_vs_back/test-color/yaw09', ['09'])
    checkdirs(base_data, 'png', 'project_front_vs_back/train-color/yaw13', 'project_front_vs_back/validate-color/yaw13', 
              'project_front_vs_back/test-color/yaw13', ['13'])
