from torchvision import datasets as dset
from torchvision import transforms as transforms

def load_dset(dataset: str, data_dir: str):
    # datasets
    if dataset.lower() == "mnist":
        transform = transforms.Compose([
            transforms.Resize(32), # (28, 28) -> (32, 32)
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5), # range: [-1.0, 1.0]
        ])
        train_dset = dset.MNIST(data_dir, train=True, transform=transform, download=True)
        val_dset = dset.MNIST(data_dir, train=False, transform=transform, download=True)
    else: # celeba
        transform = transforms.Compose([
            # transforms.CenterCrop((140, 140)),
            transforms.CenterCrop((128,128)),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # range: [-1.0, 1.0]
        ])
        train_dset = dset.CelebA(data_dir, split="train", target_type="identity", transform=transform, download=True)
        val_dset = dset.CelebA(data_dir, split="valid", target_type="identity", transform=transform, download=True)
    return train_dset, val_dset