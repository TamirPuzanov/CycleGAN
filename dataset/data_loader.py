from PIL import Image

from torch.utils.data import Dataset, DataLoader
import os, random


class Data(Dataset):
    def __init__(self, path_A, path_B, transform=None):
        super(Dataset, self).__init__()
        
        self.files_A = os.listdir(path_A)
        self.files_B = os.listdir(path_B)
        
        self.path_A = path_A
        self.path_B = path_B
        
        self.transform = transform
        
    def __len__(self):
        return len(self.files_B)
    
    def __getitem__(self, idx):
        path_A = os.path.join(self.path_A, self.files_A[random.randint(0, len(self.files_A) - 1)])
        path_B = os.path.join(self.path_B, self.files_B[idx])
        
        A = Image.open(path_A).convert("RGB")
        B = Image.open(path_B).convert("RGB")
        
        if self.transform is not None:
            A = self.transform(A)
            B = self.transform(B)
        
        return A, B


def get_loader(
    path_A, path_B, transform=None, 
    batch_size=64, num_workers=2, 
    pin_memory=True, shuffle=True
):
    return DataLoader(
        Data(path_A, path_B, transform), batch_size=batch_size, 
        shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )