import os
from os.path import isdir
import tarfile
import wget
from pathlib import Path
from PIL import Image
import torch
from torch import tensor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

_DATASET_URL = {
    "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz"}

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

# Depending on the system the code run on, the system may not support several iterations on all the dataset in only one job
# The following classnames definitions allows running the code on a sample of the dataset, to ensure a complete execution without sesison crashing

_CLASSNAMES_1 = [
    "bottle",
    "cable",
    "capsule",
    "carpet"
]

_CLASSNAMES_2 = [
    "grid",
    "hazelnut",
    "leather",
    "metal_nut"
]

_CLASSNAMES_3 = [
    "pill",
    "screw",
    "tile",
    "toothbrush"
]

_CLASSNAMES_4 = [
    "transistor",
    "wood",
    "zipper"
]

DATASETS_PATH = "C:\mvtec_anomaly_detection"

IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225])


class MVTecDataset:
    def __init__(self, cls: str, source: str = DATASETS_PATH, size: int = 224):
        """
            This constructor is used to initialized an instance of MVTecDataset. It is identified by the following parameters : 

            - cls : This string value corresponds to the class of the dataset that is considered

            - source : This string represents the path to the dataset 

            - size : This integer represents the size of the dataset 

            This __init__ method assigns the values of the parameters to the class attributes and check if the class exists in the 
            given classes names. If it is, it calls the download() method that will allow downloading the dataset content. 
            Then it creates a train and test dataset with the dataset content. 
        """
        self.cls = cls
        self.source = source
        self.size = size
        if cls in _CLASSNAMES:
            self._download(_DATASET_URL)
        self.train_ds = MVTecTrainDataset(cls, source, size)
        self.test_ds = MVTecTestDataset(cls, source, size)

    def _download(self, url_dict: dict):
        """
            This method is used to downlaod the dataset content if it is not already present in the specified source
            The parameter url_dict is a dictionnary containing the URLs for the dataset to download. 
        """
        if not isdir(self.source + "/" + self.cls):

            print("The dataset", self.cls, "is not already present in", self.source, ". Downloading", self.cls, "...")
            wget.download(url_dict[self.cls])
            with tarfile.open(self.cls + ".tar.xz") as tar:
                print("Extracting the compressed folder", self.cls, ".tar.xz....")
                tar.extractall(self.source)
            os.remove(self.cls+".tar.xz")
            print("Download of", self.cls, "in", self.source, "Complete.")    

    def get_datasets(self):
        """
            Return the train and test datasets
        """
        return self.train_ds, self.test_ds

    def get_dataloaders(self, num_workers=8, batch_size=1):
        """
            This method creating Dataloaders from the train_dataset and test_dataset
            In PyTorch, a Dataloader is a class allowing loading data from a dataset and creating an iterator over the data. 
            It allows to easily and efficiently load data 
            The Dataloader takes several parameters : batch_size, shuffle (= True if we want to have the data reshuffled at every epoch),
            num_workers that allow to parallelize calculation and make it faster, pin_memory that contributes in improving the calculation
        """
        # Creating a Dataloader from the training data
        train_dataloader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        # Creating a DataLoader from the test_dataset
        test_dataloader = DataLoader(
            self.test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        return train_dataloader, test_dataloader


class MVTecTrainDataset(ImageFolder):
    """
        This class allows the creation of a instance of MVTecTrainDataset. It takes four parameters : 

        - cls : Representing the name of the dataset class considered

        - source : String representing the path of the dataset

         - resize : Integer representing the size to resize the images (the default value is 256)

         - imagesize : Integer representing the size of the images (the default value is 224)

        In this method, several transformations to the images are performed. Resizing the image to the specified resize size, cropping the images 
        to the specified imagesize size, converting the image to a Pytorch tensor and normalizing the image with a mean and std value. 
    """
    def __init__(self, cls: str, source: str = DATASETS_PATH, resize: int = 256, imagesize: int = 224):
        super().__init__(
            root=source + "/" + cls + "/" + "train",
            transform=transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        )
        self.cls = cls
        self.size = resize


class MVTecTestDataset(ImageFolder):
    """
        This class allows the creation of a instance of MVTecTestDataset. It takes four parameters : 

        - cls : Representing the name of the dataset class considered

        - source : String representing the path of the dataset

         - resize : Integer representing the size to resize the images (the default value is 256)

         - imagesize : Integer representing the size of the images (the default value is 224)

        In this method, several transformations to the images are performed. Resizing the image to the specified resize size, cropping the images 
        to the specified imagesize size, converting the image to a Pytorch tensor and normalizing the image with a mean and std value. 

        Then the __getitem__ method is used to get a sample from the dataset. It takes only one parameter, index, an integer that represents 
        the index of the sample to be retrieved.
    """
    def __init__(self, cls: str, source: str = DATASETS_PATH, resize: int = 256, imagesize: int = 224):
        super().__init__(
            root=source + "/" + cls + "/" + "test",
            transform=transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]),
            target_transform=transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ])
        )
        self.cls = cls
        self.size = resize

    def __getitem__(self, index):
        """
            This method first gets the path of the sample and the corresponding target and then checks if the sample is 
            labeled as a "good" sample. If it is, a new image is created as a zero tensor and if it is not, it loads the target image. 
            The target image is obtained by replacing "test" with "ground_truth and replacing the file extension of the sample with _mask.png
            Then some transformations are applied to the sample and target images and finally returns the sample image, target image 
            and the class of the sample. This last value is determined by checking is the word "good" appears in the path of the sample.
            If it does, the class is set to 0, otherwise to 1.

            This method allows the dataset to return an sample in the desired format with its class and mask, making it easier to use the
            data for training and evaluating the model. 
        """

        path, _ = self.samples[index]
        sample = self.loader(path)

        if "good" in path:
            target = Image.new('RGB', (self.size, self.size))
            sample_class = 0

        else:
            target_path = path.replace("test", "ground_truth")
            target_path = target_path.replace(".png", "_mask.png")
            target = self.loader(target_path)
            sample_class = 1

        # Application of the transformations
        if self.transform is not None:
            sample = self.transform(sample)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target[:1], sample_class

