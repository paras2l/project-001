"""
Training loop for Conversational LLM
Resume-safe, Colab-compatible, fully aligned with project pipeline
Features: AMP, gradient accumulation, checkpointing, NaN safety
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

from transformer_model import TransformerModel
from dataloader import get_dataloader


# ===== CONFIGURATION =====
EPOCHS = 10
LEARNING_RATE = 3e-4
GRADIENT_ACCUMULATION_STEPS = 4
CLIP_GRADIENT = 1.0
BATCH_SIZE = 8
PAD_TOKEN_ID = 50257

# ===== PATH CONFIG (COLAB + GOOGLE DRIVE) =====
DATA_ROOT = "/content/drive/MyDrive/project 001"

DATA_FILES = [
    f"{DATA_ROOT}/tokenized_blended_skill_talk.txt",
    f"{DATA_ROOT}/tokenized_Cynaptics_persona-chat.txt",
    f"{DATA_ROOT}/tokenized_kaistlayner_empathy-dataset.txt",
    f"{DATA_ROOT}/tokenized_multiwoz.txt",
    f"{DATA_ROOT}/tokenized_OpenAssistant_oasst1.txt",
    f"{DATA_ROOT}/tokenized_ParlAI_blended_skill_talk.txt",
    f"{DATA_ROOT}/tokenized_tatsu-lab_alpaca.txt",
    f"{DATA_ROOT}/tokenized_cornell_movie_dialogs.txt",
    f"{DATA_ROOT}/tokenized_emojis.txt",
    f"{DATA_ROOT}/tokenized_wikipedia.txt",
    f"{DATA_ROOT}/tokenized_dolly.txt",
    f"{DATA_ROOT}/tokenized_flan.txt",
    f"{DATA_ROOT}/tokenized_opus_en-hi.txt",
    f"{DATA_ROOT}/tokenized_opus_en-gu.txt",
    f"{DATA_ROOT}/tokenized_opus_en-ja.txt",
    f"{DATA_ROOT}/tokenized_opus_en-es.txt",
    f"{DATA_ROOT}/tokenized_opus_en-ko.txt",
]

CHECKPOINT_DIR = f"{DATA_ROOT}/checkpoint"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/latest.pt"

import os
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# AMP scaler for GPU speed boost (30-50% faster)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))


def train():
    # ===== INITIALIZE MODEL =====
    print("Initializing model...")
    model = TransformerModel(
        vocab_size=50268,
        embed_dim=384,
        num_heads=6,
        num_layers=8,
        dropout=0.1
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # ===== OPTIMIZER & SCHEDULER =====
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    
    # ===== LOAD DATA =====
    print("Loading dataset...")
    loader = get_dataloader(DATA_FILES, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Total batches per epoch: {len(loader)}")
    
# Learning rate scheduler with gradient accumulation correction
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(len(loader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
    )
    
    # ===== RESUME SUPPORT =====
    start_epoch = 0
    global_step = 0
    ckpt = None
    
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        except Exception as e:
            print("Checkpoint corrupted, skipping load:", e)
    
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["step"]
        print(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # ===== TRAINING LOOP =====
    model.train()
    print("\nStarting training...")
    
    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        total_loss = 0.0
        
        progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}")
        
        for step, (input_ids, targets, attention_mask) in progress_bar:
            
            # Move to device
            input_ids = input_ids.to(DEVICE)
            targets = targets.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            
            # Forward pass with AMP autocast
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(input_ids, attention_mask)
                
                vocab_size = logits.shape[-1]
                
                # Calculate loss with contiguous memory layout (AMP safe)
                loss = loss_fn(
                    logits.contiguous().view(-1, vocab_size),
                    targets.contiguous().view(-1)
                )
            
            # Loss explosion safety
            if torch.isnan(loss):
                print("NaN detected! Skipping batch...")
                continue
            
            # Scale for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            
            # Update weights every N steps
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                
                # Gradient clipping with AMP
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRADIENT)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                scheduler.step()
                global_step += 1
            
            # Track loss (unscale for display)
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{total_loss / (step + 1):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                "step": global_step
            })
            
            # Save checkpoint every 200 steps (atomic save to prevent corruption)
            if global_step > 0 and global_step % 200 == 0:
                temp_path = CHECKPOINT_PATH + ".tmp"

                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                }, temp_path)

                os.replace(temp_path, CHECKPOINT_PATH)
        
        # End of epoch stats
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save both: latest (for resume) + epoch copy (for history backup)
        checkpoint_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "step": global_step,
            "avg_loss": avg_loss
        }
        
        # 1. Overwrite latest checkpoint for resume (atomic save)
        temp_path = CHECKPOINT_PATH + ".tmp"
        torch.save(checkpoint_data, temp_path)
        os.replace(temp_path, CHECKPOINT_PATH)
        print(f"Latest checkpoint saved: {CHECKPOINT_PATH}")
        
        # 2. Save unique epoch checkpoint for backup history (atomic save)
        epoch_checkpoint_path = f"{CHECKPOINT_DIR}/epoch_{epoch+1}.pt"
        temp_epoch_path = epoch_checkpoint_path + ".tmp"
        torch.save(checkpoint_data, temp_epoch_path)
        os.replace(temp_epoch_path, epoch_checkpoint_path)
        print(f"Epoch checkpoint saved: {epoch_checkpoint_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    train()