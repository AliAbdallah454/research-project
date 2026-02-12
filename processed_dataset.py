import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# ImageNet mean and std
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

train_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

val_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

class ProcessedDataset(Dataset):
    def __init__(self, image_paths, targets, transform, out_size=(360, 640)):

        self.image_paths = list(image_paths)
        self.targets = list(targets)
        self.out_h, self.out_w = out_size

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)

        y = torch.tensor(self.targets[idx], dtype=torch.float32).clone()    

        return img, y