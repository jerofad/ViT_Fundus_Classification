""" This file is used for creating image subfolder
    based on labels. Used for the o_O solution repo

"""
import os
import shutil
import pandas as pd
import numpy as np

csv_path = "/home/jerryfad/RetinaClassification/datasets/eyepacs/trainLabels.csv"
data_df = pd.read_csv(csv_path)

class_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']

def change_label(x):
    return class_labels[x]


def train_validate_test_split(df, train_percent=.6, validate_percent=.2, 
                                seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]

    return train, validate, test


def copy_data(df, split):

    org_dir = "/home/jerryfad/RetinaClassification/datasets/eyepacs/train"
    labels = df.sort_values('level')
    class_names = list(labels.level.unique())
    # create split directory name

    os.makedirs(split)

    # create directory for each class_name
    for i in class_names:
        os.makedirs(os.path.join(split,i))
    
    for c in class_names:
        for i in list(labels[labels['level']==c]['image']):

            #create path to the image 
            orig_img_path = os.path.join(org_dir, i+'.png') #?

            #If image has not already exist in the new folder create one        
            if not os.path.exists(split+'/'+c+i):
                # move the image 
                move_image_to_cat = shutil.copy2(orig_img_path, split+'/'+c)    


data_df['Class'] = data_df['level'].apply(change_label)
train, validate, test = train_validate_test_split(data_df)


copy_data(train, "train")
copy_data(test, "test")
copy_data(validate, "val")


# base path
base_path = "/home/jerryfad/RetinaClassification/"
# list of directories we want to move.
dir_list = ["train", "test", "val"]
  
# path to destination directory
dest = "/home/jerryfad/RetinaClassification/datasets/EyePacs_new/"
  
print("Before moving directories:")
print(os.listdir(base_path))
  
# traverse each directory in dir_list
for dir_ in dir_list:
  
    # create path to the directory in the
    # dir_list.
    source = os.path.join(base_path, dir_)
  
    # check if it is an existing directory
    if os.path.isdir(source):
  
        # move to destination path
        shutil.move(source, dest)
  
print("After moving directories:")
print(os.listdir(base_path))



