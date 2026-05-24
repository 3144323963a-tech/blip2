
"""
Flickr8k 数据加载模块。
读取前 200 张图片及对应 caption，返回 DataLoader。
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd



class Flickr8kDataset(Dataset):
    """每张图片 × 5 条 caption，每个 (image, caption) 是独立样本。"""

    def __init__(self, data_dir, captions_file, split="train"):
        self.data_dir = data_dir

        # 只取 images 目录下实际存在的图片
        available = set(os.listdir(os.path.join(data_dir, "images")))
        df = pd.read_csv(captions_file)
        df = df[df["image"].isin(available)]
        unique_images = sorted(df["image"].unique())[:200]

        n = len(unique_images)
        if split == "train":
            allowed = set(unique_images[:int(n * 0.8)])
        elif split == "val":
            allowed = set(unique_images[int(n * 0.8):])
        else:
            allowed = set(unique_images)

        df = df[df["image"].isin(allowed)]
        self.samples = list(zip(df["image"].values, df["caption"].values))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        img_path = os.path.join(self.data_dir, "images", img_name)
        image = Image.open(img_path).convert("RGB")
        return image, caption


def collate_fn(batch, clip_processor, opt_tokenizer, max_length=32):
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    clip_inputs = clip_processor(images=images, return_tensors="pt")

    text_inputs = opt_tokenizer(
        captions, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    )

    return {
        "pixel_values": clip_inputs["pixel_values"],
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "labels": text_inputs["input_ids"].clone(),
    }


def create_dataloaders(data_dir, captions_file, clip_processor, opt_tokenizer,
                       batch_size=8, num_workers=0):
    train_ds = Flickr8kDataset(data_dir, captions_file, "train")
    val_ds = Flickr8kDataset(data_dir, captions_file, "val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, clip_processor, opt_tokenizer),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, clip_processor, opt_tokenizer),
    )
    return train_loader, val_loader
