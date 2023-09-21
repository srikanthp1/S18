import sys 

import warnings

# Suppress all user warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from PIL import Image
import shutil
from urllib.request import urlretrieve
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from torchvision import transforms

cudnn.benchmark = True

from sklearn.model_selection import train_test_split


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        print("Dataset already exists on the disk. Skipping download.")
        return

    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(filepath)) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    shutil.unpack_archive(filepath, extract_dir)


def merge_trainval_test(filepath):
    """
        #   Image CLASS-ID SPECIES BREED ID
        #   ID: 1:37 Class ids
        #   SPECIES: 1:Cat 2:Dog
        #   BREED ID: 1-25:Cat 1:12:Dog
        #   All images with 1st letter as captial are cat images
        #   images with small first letter are dog images
    """
    merge_dir = os.path.dirname(os.path.abspath(f'{filepath}/annotations/data.txt'))
    #if os.path.exists(merge_dir):
    #    print("Merged data is already exists on the disk. Skipping creating new data file.")
    #    return
    df = pd.read_csv(f"{filepath}/annotations/trainval.txt", sep=" ", 
                     names=["Image", "ID", "SPECIES", "BREED ID"])
    df2 = pd.read_csv(f"{filepath}/annotations/test.txt", sep=" ",
                      names=["Image", "ID", "SPECIES", "BREED ID"])
    frame = [df, df2]
    df = pd.concat(frame)
    df.reset_index(drop=True)
    df.to_csv(f'{filepath}/annotations/data.txt', index=None, sep=' ')
    print("Merged data is created.")


dataset_directory = os.path.join("/kaggle/working/dataset")
# os.mkdir(dataset_directory)

filepath = os.path.join(dataset_directory, "images.tar.gz")
# download_url(
#     url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", filepath=filepath,
# )
# extract_archive(filepath)

filepath = os.path.join(dataset_directory, "annotations.tar.gz")
# download_url(
#     url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", filepath=filepath,
# )
# extract_archive(filepath)

filepath = os.path.join(dataset_directory)
merge_trainval_test(filepath)

dataset = pd.read_csv(f"{filepath}/annotations/data.txt", sep=" ")

image_ids = []
labels = []
with open(f"{filepath}/annotations/trainval.txt") as file:
    for line in file:
        image_id, label, *_ = line.strip().split()
        image_ids.append(image_id)
        labels.append(int(label)-1)

classes = [
    " ".join(part.title() for part in raw_cls.split("_"))
    for raw_cls, _ in sorted(
        {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, labels)},
        key=lambda image_id_and_label: image_id_and_label[1],
    )
    ]

idx_to_class = dict(zip(range(len(classes)), classes))

dataset['nID'] = dataset['ID'] - 1
decode_map = idx_to_class
def decode_label(label):
    return decode_map[int(label)]
dataset["class"] = dataset["nID"].apply(lambda x: decode_label(x))


y = dataset['class']
x = dataset['Image']

trainval, x_test, y_trainval, y_test = train_test_split(x, y,
                                                        stratify=y, 
                                                        test_size=0.2,
                                                        random_state=42)

x_train, x_val, y_train, y_val = train_test_split(  trainval, y_trainval,
                                                    stratify=y_trainval, 
                                                    test_size=0.3,
                                                    random_state=42)

root_directory = os.path.join(dataset_directory)
images_directory = os.path.join(root_directory, "images")
masks_directory = os.path.join(root_directory, "annotations", "trimaps")

train_images_filenames = x_train.reset_index(drop=True)
val_images_filenames = x_val.reset_index(drop=True)
test_images_filenames = x_test.reset_index(drop=True)


class OxfordPetDataset(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform=None, transform_mask=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames.loc[idx] + '.jpg' 
        image = Image.open(os.path.join(self.images_directory, image_filename)).convert('RGB')
        mask = Image.open(
            os.path.join(self.masks_directory, image_filename.replace(".jpg", ".png")))
        #mask = preprocess_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image)
            transformed_m = self.transform_mask(mask)
            image = transformed
            mask = transformed_m
        return image, mask
    

train_transform = transforms.Compose([transforms.Resize((240, 240)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
target_transform = transforms.Compose([transforms.PILToTensor(),     
                                       transforms.Resize((240, 240)),     
                                       transforms.Lambda(lambda x: (x-1).squeeze().type(torch.LongTensor)) ])


train_dataset = OxfordPetDataset(train_images_filenames, 
                                 images_directory, 
                                 masks_directory, 
                                 transform=train_transform, 
                                 transform_mask=target_transform)


val_dataset = OxfordPetDataset(val_images_filenames,
                               images_directory,
                               masks_directory,
                               transform=train_transform,
                               transform_mask=target_transform)

params = {
    "device": "cuda",
    "lr": 0.001,
    "batch_size": 32,
    "num_workers": 2,
    "epochs": 25,
}

train_loader = DataLoader(
    train_dataset,
    batch_size=params["batch_size"],
    shuffle=True,
    num_workers=params["num_workers"],
    pin_memory=False,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=params["batch_size"],
    shuffle=True,
    num_workers=params["num_workers"],
    pin_memory=False,
)

def preprocess_mask(mask):
    mask = np.float32(mask) / 255
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask

import torch.nn.functional as F

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, input, target):
        # Convert class labels to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Calculate Dice coefficient for each class
        dice_coeffs = []
        for class_idx in range(self.num_classes):
            intersection = torch.sum(input[:, class_idx, :, :] * target_one_hot[:, class_idx, :, :])
            union = torch.sum(input[:, class_idx, :, :] + target_one_hot[:, class_idx, :, :])
            dice_coeffs.append((2.0 * intersection + 1e-5) / (union + 1e-5))

        # Calculate the average Dice loss
        dice_loss = 1.0 - torch.mean(torch.stack(dice_coeffs))

        return dice_loss


from model import UNet

encode_arg = sys.argv[1]

decode_arg = sys.argv[2]

loss_func = sys.argv[3]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(3, 3, encode_arg, decode_arg).to(device)

if loss_func == 'bce':
    criterion = nn.CrossEntropyLoss(ignore_index=255)
elif loss_func == 'diceloss':
    criterion = MultiClassDiceLoss(3)
# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 25

for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    i=0
    for data in train_loader:
        # print(data)
        
        images, masks = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        i=i+1
        print('batch number:', str(i), ' : ', loss.item())
            
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
    

# python main.py maxpool transpose bce
# python main.py maxpool transpose diceloss
# python main.py stridedconv transpose bce
# python main.py stridedconv upsample diceloss