import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Compose, \
    RandomResizedCrop, RandomHorizontalFlip
from torch.utils.data import random_split, DataLoader
import torchvision

def get_cnmc(data_path: str = './data/CNMC'):

    tr = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainset = ImageFolder(data_path + '/train', transform=tr)
    testset = ImageFolder(data_path + '/val', transform=tr)

    print (f"Trainset: {len(trainset)}")

    # Print one sample from the dataset trainset
    print(trainset[0])

    # Print the number of classes in the dataset
    print(f"Number of classes in the dataset: {len(trainset.classes)}")

    # Print the classes in the dataset
    print(f"Classes in the dataset: {trainset.classes}")

    return trainset, testset


""" We are doing IID Partitioning for simplicity"""
def prepare_dataset(num_partitions: int, 
                    batch_size: int, 
                    val_ratio: float = 0.1):
    trainset, testset = get_cnmc()
    
   # split the trainset into 'num_partitions' trainset
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * (num_partitions - 1)
    partition_len.append(len(trainset) - sum(partition_len))

    trainsets = random_split(trainset, partition_len, generator=torch.Generator().manual_seed(2023))

    # create dataloader with train+val support
    trainloaders = []
    valloaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(num_total * val_ratio)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, [num_train, num_val], torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))


    testloader = DataLoader(testset, batch_size=128)

    # print the number of trainloaders
    print(f"Number of trainloaders: {len(trainloaders)}")

    # print the number of valloaders
    print(f"Number of valloaders: {len(valloaders)}")

    # print the number of testloaders
    print(f"Number of testloaders: {len(testloader)}")

    return trainloaders, valloaders, testloader





