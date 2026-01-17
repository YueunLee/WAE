# import torch
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader, random_split, Subset, Dataset
# from torchvision import datasets, transforms
# import os
# from PIL import Image

# class CelebADataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.root = root
#         self.transform = transform
#         self.paths = [
#             os.path.join(root, f) 
#             for f in os.listdir(root) 
#             if f.lower().endswith(('.jpg', '.jpeg', '.png'))
#         ]

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         img = Image.open(self.paths[idx]).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         return img

# class CelebADataModule(pl.LightningDataModule):
#     def __init__(self, config):
#         super().__init__()
#         self.cfg = config 
#         pl.seed_everything(self.cfg['SEED'])
        
#         self.data_dir = self.cfg['DATA_DIR'] + f"_resized_{self.cfg['IMAGE_SIZE']}"

#         self.train_transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])

#         self.val_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])

#     def setup(self, stage=None):
#         full_dataset = CelebADataset(root=self.data_dir)
        
#         n_train = int(len(full_dataset) * 0.9)
#         n_val = len(full_dataset) - n_train

#         generator = torch.Generator().manual_seed(self.cfg['SEED'])
        
#         train_ds_dummy, val_ds_dummy = random_split(
#             full_dataset, [n_train, n_val], generator=generator
#         )

#         train_root_ds = CelebADataset(root=self.data_dir, transform=self.train_transform)
#         self.train_dataset = Subset(train_root_ds, train_ds_dummy.indices)

#         val_root_ds = CelebADataset(root=self.data_dir, transform=self.val_transform)
#         self.val_dataset = Subset(val_root_ds, val_ds_dummy.indices)

#         print(f"Loaded Preprocessed Data from: {self.data_dir}")

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.cfg['BATCH_SIZE'],
#             shuffle=True,
#             num_workers=self.cfg['NUM_WORKERS'],
#             pin_memory=True,
#             drop_last=True,
#             persistent_workers=True
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.cfg['BATCH_SIZE'],
#             shuffle=False,
#             num_workers=self.cfg['NUM_WORKERS'],
#             pin_memory=True,
#             drop_last=True,
#             persistent_workers=True
#         )

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import datasets, transforms
import os
from PIL import Image
from tqdm import tqdm

class CelebADataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.paths = [
            os.path.join(root, f) 
            for f in os.listdir(root) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

class CelebADataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config 
        pl.seed_everything(self.cfg['SEED'])
        
        self.data_dir = self.cfg['DATA_DIR'] + f"_resized_{self.cfg['IMAGE_SIZE']}"

        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def prepare_data(self):
        if os.path.exists(self.data_dir) and len(os.listdir(self.data_dir)) > 0:
            return

        print(f"Preprocessing images from {self.cfg['DATA_DIR']} to {self.data_dir}...")
        os.makedirs(self.data_dir, exist_ok=True)

        pre_transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize((self.cfg['IMAGE_SIZE'], self.cfg['IMAGE_SIZE']))
        ])

        files = [f for f in os.listdir(self.cfg['DATA_DIR']) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for fname in tqdm(files):
            src_path = os.path.join(self.cfg['DATA_DIR'], fname)
            dst_path = os.path.join(self.data_dir, fname)
            
            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")
                    img_processed = pre_transform(img)
                    img_processed.save(dst_path, quality=95)
            except Exception as e:
                print(f"Error processing {fname}: {e}")

    def setup(self, stage=None):
        full_dataset = CelebADataset(root=self.data_dir)
        
        n_train = int(len(full_dataset) * 0.9)
        n_val = len(full_dataset) - n_train

        generator = torch.Generator().manual_seed(self.cfg['SEED'])
        
        train_ds_dummy, val_ds_dummy = random_split(
            full_dataset, [n_train, n_val], generator=generator
        )

        train_root_ds = CelebADataset(root=self.data_dir, transform=self.train_transform)
        self.train_dataset = Subset(train_root_ds, train_ds_dummy.indices)

        val_root_ds = CelebADataset(root=self.data_dir, transform=self.val_transform)
        self.val_dataset = Subset(val_root_ds, val_ds_dummy.indices)

        print(f"Loaded Preprocessed Data from: {self.data_dir}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg['BATCH_SIZE'],
            shuffle=True,
            num_workers=self.cfg['NUM_WORKERS'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg['BATCH_SIZE'],
            shuffle=False,
            num_workers=self.cfg['NUM_WORKERS'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )