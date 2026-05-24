"""
Mini-BLIP2 训练脚本。
冻结 CLIP ViT 和 OPT-125M，只训练 Mini Q-Former + Projection。
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import CLIPProcessor, AutoTokenizer
from tqdm import tqdm

# 把 code 目录加入 path，方便 import dataset 和 model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import create_dataloaders
from model import MiniBLIP2


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- 1. 加载 CLIP Processor 和 OPT Tokenizer ----
    print("Loading CLIP processor and OPT tokenizer ...")
    clip_processor = CLIPProcessor.from_pretrained(config["vision_model"])
    opt_tokenizer = AutoTokenizer.from_pretrained(config["language_model"])
    if opt_tokenizer.pad_token is None:
        opt_tokenizer.pad_token = opt_tokenizer.eos_token

    # ---- 2. 创建 DataLoader ----
    data_dir = config["data_dir"]
    captions_file = config["captions_file"]
    # 路径处理：支持绝对路径和相对于 code/.. 的路径
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(__file__), "..", data_dir)
    if not os.path.isabs(captions_file):
        captions_file = os.path.join(os.path.dirname(__file__), "..", captions_file)

    print(f"Data dir: {os.path.abspath(data_dir)}")
    print(f"Captions file: {os.path.abspath(captions_file)}")

    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        captions_file=captions_file,
        clip_processor=clip_processor,
        opt_tokenizer=opt_tokenizer,
        batch_size=config["batch_size"],
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- 3. 创建模型 ----
    print("Building model ...")
    model = MiniBLIP2(
        vision_model_name=config["vision_model"],
        language_model_name=config["language_model"],
        num_queries=config["num_queries"],
        qformer_hidden=config["qformer_hidden"],
        qformer_layers=config["qformer_layers"],
        qformer_heads=config["qformer_heads"],
    ).to(device)

    # 确认冻结/可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.1f}%)")

    # ---- 4. 优化器 & 调度器 ----
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # ---- 5. 训练循环 ----
    loss_history = {"train": [], "val": []}
    best_val_loss = float("inf")
    output_dir = config["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), "..", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(config["epochs"]):
        # --- Train ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [train]")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values, input_ids, attention_mask, labels)
            loss = outputs["loss"]
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        loss_history["train"].append(avg_train_loss)

        # --- Val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [val]"):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(pixel_values, input_ids, attention_mask, labels)
                val_loss += outputs["loss"].item()

        avg_val_loss = val_loss / len(val_loader)
        loss_history["val"].append(avg_val_loss)
        scheduler.step()

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        # --- Save checkpoint ---
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "config": config,
        }
        torch.save(checkpoint, os.path.join(output_dir, "last_checkpoint.pt"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(output_dir, "best_checkpoint.pt"))
            print(f"  -> Best checkpoint saved (val_loss={best_val_loss:.4f})")

    # ---- 6. 保存 loss 曲线 ----
    with open(os.path.join(output_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f, indent=2)
    print(f"Training done. Best val loss: {best_val_loss:.4f}")
    print(f"Output saved to: {os.path.abspath(output_dir)}")

    return model, loss_history


if __name__ == "__main__":
    import os as _os
    _base = _os.path.join(_os.path.dirname(__file__), "..", "models")
    config = {
        # 模型（本地路径）
        "vision_model": _os.path.join(_base, "clip-vit-base-patch32"),
        "language_model": _os.path.join(_base, "opt-125m"),
        # Q-Former 配置
        "num_queries": 16,
        "qformer_hidden": 512,
        "qformer_layers": 2,
        "qformer_heads": 8,
        # 训练
        "epochs": 30,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        # 数据路径（相对于 code/.. = 项目根目录）
        "data_dir": "data",
        "captions_file": "data/captions.txt",
        # 输出
        "output_dir": "checkpoints",
    }

    train(config)
