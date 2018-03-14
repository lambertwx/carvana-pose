# -*- coding: utf-8 -*-
"""
Experiments with uploading very small amounts of data for learning.

Created on Fri Jan 19 22:51:54 2018

@author: lambert.wixson
"""

from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
from skimage.io import imread_collection, imread

#%%
app = ClarifaiApp(api_key = 'REPLACETHISWITHYOUROWN')

#%%

def upload(folder : str, num_to_skip, num_to_load, app, concepts):
    coll = imread_collection(folder, conserve_memory=True, check_files=False)
    for i, img in enumerate(coll):
        if i < num_to_skip: 
            continue
        elif i < (num_to_skip + num_to_load):
            print("{0}: {1}".format(i, coll.files[i]))
            climg = app.inputs.create_image_from_filename(coll.files[i], concepts=concepts)
        else:
            break
        
    return coll
            
#%%
if False:
    # Experiments with learning from very small amounts of uploaded data.
    coll0 = upload("C:/nobak/kaggle_carvana/project_front_vs_back/train-color/yaw01/*.png", 0, 100, app, ["yaw-0"])
    coll180 = upload("C:/nobak/kaggle_carvana/project_front_vs_back/train-color/yaw09/*.png", 0, 100, app, ["yaw-180"])
    coll90 = upload("C:/nobak/kaggle_carvana/project_front_vs_back/train-color/yaw05/*.png", 0, 100, app, ["yaw-90"])
    coll270 = upload("C:/nobak/kaggle_carvana/project_front_vs_back/train-color/yaw13/*.png", 0, 100, app, ["yaw-270"])
    
    # The above, when I trained the model, produced good classifications for 0 and 180, but for 90 vs 270 it
    # was little better than chance. ROC AUC for 270 was .858 and for 90 it was .845.  
    # So I added a total of 500 additional training images each for 90 and 270.
    coll270 = upload("C:/nobak/kaggle_carvana/project_front_vs_back/train-color/yaw13/*.png", 100, 400, app, ["yaw-270"])
    coll90 = upload("C:/nobak/kaggle_carvana/project_front_vs_back/train-color/yaw05/*.png", 100, 400, app, ["yaw-90"])
    # As a result of adding the additional data, the stats got better AUC for 270 was .903, and for 90 it was .900.  
    # But there is still plenty of errors.
    coll90 = upload("C:/nobak/kaggle_carvana/project_front_vs_back/train-color/yaw05/*.png", 500, 500, app, ["yaw-90"])
    coll270 = upload("C:/nobak/kaggle_carvana/project_front_vs_back/train-color/yaw13/*.png", 500, 500, app, ["yaw-270"])
    # Now that we have 1000 examples each for 90 and 270, their behavior is getting better, with ROC of .937 and 
    # .936 respectively.  Still the probability of a wrong classification given that yaw-270 is predicted is .282, 
    # and the prob given that yaw-90 is predicted is .214.  
    # So the predictor still has roughly a 25% chance of being wrong.  That's not great.
    # Try adding another 1000
    coll90 = upload("C:/nobak/kaggle_carvana/project_front_vs_back/train-color/yaw05/*.png", 1000, 1000, app, ["yaw-90"])
    coll270 = upload("C:/nobak/kaggle_carvana/project_front_vs_back/train-color/yaw13/*.png", 1000, 1000, app, ["yaw-270"])
    # Now that we have 2000 samples of each, the ROC is .953 for both 90 and 270.  The prob of a wrong classification given that 
    # yaw-270 is predicted is .229, and the prob given that yaw-90 is predicted is .163.  So, some improvement, but 
    # still not great.  Needs more data still.
    
    # Now try loading another 2000 samples.  The uploads failed on image 2160 of coll90, because I exceeded
    # my total of 5000 operations for the month.  Apparently each image upload counts as an operation.  I'll have
    # to try again after 2/11/18, I guess.
    coll90 = upload("C:/nobak/kaggle_carvana/project_front_vs_back/train-color/yaw05/*.png", 2000, 2000, app, ["yaw-90"])
    coll270 = upload("C:/nobak/kaggle_carvana/project_front_vs_back/train-color/yaw13/*.png", 2000, 2000, app, ["yaw-270"])
    
   # foo = app.inputs.create_image_from_filename(coll0.files[0], concepts=["yaw-0"])
    