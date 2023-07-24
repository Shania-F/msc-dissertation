import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models, datasets
from PIL import Image
import matplotlib.pyplot as plt


class PoseDataset(Dataset):
    def __init__(self, dataset_path, tfs, train=True):
        self.dataset_path = dataset_path
        self.train = train
        self.tfs = tfs
        self.image_paths, self.poses = self.__read_data__()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        pose = self.poses[idx]
        # Load the image and preprocess it
        image = Image.open(image_path)
        image = self.tfs(image)
        # Convert the pose to a tensor
        pose_tensor = torch.tensor(pose, dtype=torch.float32)
        return image, pose_tensor

    def __read_data__(self):
        image_paths = []
        poses = []
        # dataset_path -> ./datasets/KingsCollege
        file_name = 'dataset_train.txt' if self.train else 'dataset_test.txt'
        with open("/".join([self.dataset_path, file_name])) as f:
            lines = f.readlines()
        for i in range(3, len(lines)):  # skipping first 3 lines
            data = lines[i].split()
            image_paths.append("/".join([self.dataset_path, data[0]]))
            poses.append([float(x) for x in data[1:]])

        return image_paths, poses


def get_dataloader(train=True):  # dataset_path, batch_size,
    if train:
        transform = transforms.Compose([
            transforms.Resize(256),  # 300, 299
            transforms.RandomCrop(224),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_data = PoseDataset("../datasets/KingsCollege", train=True, tfs=transform)
        dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=4)
    else:  # test
        transform = transforms.Compose([
            transforms.Resize(256),  # 300, 299
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_data = PoseDataset("../datasets/KingsCollege", train=False, tfs=transform)
        dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=False, num_workers=4)

    return dataloader


# test it (this doesn't run when imported; only when run directly)
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    d1 = PoseDataset("../datasets/KingsCollege", tfs=transform)
    x, y = d1.__getitem__(25)
    print(x.shape)
    print(y)

    train_loader = get_dataloader(train=True)
    print(f"Length of train_dataloader: {len(train_loader)} batches of {32}...")

    train_features_batch, train_labels_batch = next(iter(train_loader))
    print(train_features_batch.shape, train_labels_batch.shape)
    img, pose = train_features_batch[0], train_labels_batch[0]

    plt.imshow(img.squeeze())
    # img.permute(1, 2, 0)
    plt.axis(False)
    print(f"Image size: {img.shape}")
    print(f"Label: {pose}, label size: {pose.shape}")
