import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transform(train: bool = True) -> transforms.Compose:
    
    if train:
        return transforms.Compose([
            transforms.Resize(256),                  
            transforms.RandomCrop(224),              
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.ToTensor(),                   
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),              
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class FlickrDataset(Dataset):
    
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        transform,
        max_length: int = 77,
        caption_idx: int = 0,
    ):
        self.data        = hf_dataset
        self.tokenizer   = tokenizer
        self.transform   = transform
        self.max_length  = max_length
        self.caption_idx = caption_idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]

        
        image = item["image"]
        if image.mode != "RGB":
            
            image = image.convert("RGB")
        image = self.transform(image)   

        
        captions = item["caption"]
        if isinstance(captions, list):
            caption = captions[self.caption_idx]
        else:
            caption = captions  

        
        encoded = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        
        tokens = {
            "input_ids":      encoded["input_ids"].squeeze(0),       
            "attention_mask": encoded["attention_mask"].squeeze(0),   
        }

        return image, tokens


def build_loaders(
    train_ds: FlickrDataset,
    val_ds:   FlickrDataset,
    test_ds:  FlickrDataset,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple:
   
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,          
        num_workers=num_workers,
        pin_memory=True,        
        drop_last=True,        
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,          
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
