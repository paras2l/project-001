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

# Checkpoint path (change to Google Drive path on Colab)
CHECKPOINT_PATH = "latest_checkpoint.pt"

# Dataset files (matching your project)
DATA_FILES = [
    "tokenized_blended_skill_talk.txt",
    "tokenized_Cynaptics_persona-chat.txt",
    "tokenized_kaistlayner_empathy-dataset.txt",
    "tokenized_multiwoz.txt",
    "tokenized_OpenAssistant_oasst1.txt",
    "tokenized_ParlAI_blended_skill_talk.txt",
    "tokenized_tatsu-lab_alpaca.txt",
    "tokenized_cornell_movie_dialogs.txt",
    "tokenized_emojis.txt",
    "tokenized_wikipedia.txt",
    "tokenized_squad.txt",
    "tokenized_dolly.txt",
    "tokenized_flan.txt",
]

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# AMP scaler for GPU speed boost (30-50% faster)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))


def train():
    # ===== INITIALIZE MODEL =====
    print("Initializing model...")
    model = TransformerModel(
        vocab_size=50262,
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
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
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
                
                # Calculate loss using reshape (safer than view)
                loss = loss_fn(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1)
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
            
            # Save checkpoint every 200 steps
            if global_step > 0 and global_step % 200 == 0:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                }, CHECKPOINT_PATH)
        
        # End of epoch stats
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint at end of each epoch
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "step": global_step,
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved: {CHECKPOINT_PATH}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    train()