import os
import cv2
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, models, transforms

os.chdir('/Users/joeholt/Documents/College Local/Current/CS 639/proj')

def process_img(path, new_size, threshold=0.4):
    img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (new_size[0],  new_size[1]))
    # img = cv2.threshold(img, 255*threshold, 255, cv2.THRESH_BINARY)[1]
    return img
    
def class_label(string):
    dic = {
        'palm': 0,
        'l': 1,
        'fist': 2,
        'fist_moved': 3,
        'thumb': 4,
        'index': 5,
        'ok': 6,
        'palm_moved': 7,
        'c': 8,
        'down': 9
    }
    return dic[string]

def load_dataset_pytorch():
    #https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    os.chdir('/Users/joeholt/Documents/College Local/Current/CS 639/proj/test') 
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/gesture_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    print(image_datasets['train'].class_to_idx)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, class_names, dataset_sizes

def load_dataset(transformed_size=(226, 226)):
    """
    Loads dataset and returns X, y. Loads dataset from leapGestRecog
    data store in this project
    
    Returns:
    X representing processed dataset images in arrays
    y representing classes as strings for those items
    """
    os.chdir('/Users/joeholt/Documents/College Local/Current/CS 639/proj')
    root_data_dir = os.path.join(os.getcwd(), 'data', 'leapGestRecog')
    
    y = []
    X = []
    
    # get data
    for subject_dir in os.listdir(root_data_dir):
        if subject_dir == '.ipynb_checkpoints':
            continue
        
        subject_path = os.path.join(root_data_dir, subject_dir)
        
        for class_dir in os.listdir(subject_path):
            
            class_path = os.path.join(subject_path, class_dir)
            class_name = class_dir.split('_')[1]
            
            for filename in os.listdir(class_path):
                
                if filename.split('.')[1] != 'png':
                    continue
                
                full_path = os.path.join(class_path, filename)
                img = torch.tensor(process_img(full_path, transformed_size[::-1])).float().permute(2, 0, 1)
                class_index = torch.tensor(class_label(class_name)).long()
                X.append(img)
                y.append(class_index)
    
    # convert to numpy
    X = torch.stack(X)
    y = torch.stack(y)

    return TensorDataset(X, y)