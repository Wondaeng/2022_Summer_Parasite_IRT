from torchvision import datasets, transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, Subset
import random


def custom_dataset_folder(data_pth):
    data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                          ])

    image_datasets = datasets.ImageFolder(data_pth, data_transforms)
    print('Folders are assigned to indices as:', image_datasets.class_to_idx)
    return image_datasets


def split_train_test(dataset, split_ratio=0.7, pos_neg_ratio=1, batch_size = 64):

    pos_index, neg_index = [], []
    for i, j in enumerate(dataset):
        if j[1] == 0:
            neg_index.append(i)
        else:
            pos_index.append(i)

    neg_index_balanced = random.sample(neg_index, len(pos_index)*pos_neg_ratio)
    print(len(pos_index))
    print(len(neg_index_balanced))
    indices = pos_index + neg_index_balanced

    dataset_sub = Subset(dataset, indices)

    train_size = int(split_ratio * len(dataset_sub))
    test_size = len(dataset_sub) - train_size
    train_dataset, test_dataset = random_split(dataset_sub, [train_size, test_size])

    print(f'#train data: {len(train_dataset)}')
    print(f'#test data: {len(test_dataset)}')
    print(f'#total data: {len(train_dataset) + len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader
