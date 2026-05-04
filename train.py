"""
Training loop for Conversational LLM
Resume-safe, Colab/Kaggle/Local compatible
Features: AMP, gradient accumulation, atomic checkpointing, cross platform resume
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import json

from transformer_model import TransformerModel
from dataloader import get_dataloader


# ===== CONFIGURATION =====
EPOCHS = 10
LEARNING_RATE = 3e-4
GRADIENT_ACCUMULATION_STEPS = 4
CLIP_GRADIENT = 1.0
BATCH_SIZE = 8
PAD_TOKEN_ID = 50257


# ===== ENVIRONMENT AUTO DETECTION =====
def get_env():
    if os.path.exists("/kaggle"):
        return "kaggle"
    elif os.path.exists("/content"):
        return "colab"
    else:
        return "local"

ENV = get_env()

# Read only dataset paths
if ENV == "kaggle":
    DATA_ROOT = "/kaggle/input/datasets/paraslashkari/project001"
elif ENV == "colab":
    DATA_ROOT = "/content/drive/MyDrive/project 001"
else:
    DATA_ROOT = "./data"

# Writable checkpoint paths
if ENV == "kaggle":
    CHECKPOINT_DIR = "/kaggle/working/checkpoint"
elif ENV == "colab":
    CHECKPOINT_DIR = "/content/drive/MyDrive/project 001/checkpoint"
else:
    CHECKPOINT_DIR = "./checkpoint"
LATEST_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")
RESUME_INFO_PATH = os.path.join(CHECKPOINT_DIR, "resume.json")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"ENV: {ENV}")
print(f"Checkpoint dir: {CHECKPOINT_DIR}")


# ===== DATASET FILES =====
DATA_FILES = [
    os.path.join(DATA_ROOT, "tokenized_blended_skill_talk.txt"),
    os.path.join(DATA_ROOT, "tokenized_Cynaptics_persona-chat.txt"),
    os.path.join(DATA_ROOT, "tokenized_kaistlayner_empathy-dataset.txt"),
    os.path.join(DATA_ROOT, "tokenized_multiwoz.txt"),
    os.path.join(DATA_ROOT, "tokenized_OpenAssistant_oasst1.txt"),
    os.path.join(DATA_ROOT, "tokenized_ParlAI_blended_skill_talk.txt"),
    os.path.join(DATA_ROOT, "tokenized_tatsu-lab_alpaca.txt"),
    os.path.join(DATA_ROOT, "tokenized_cornell_movie_dialogs.txt"),
    os.path.join(DATA_ROOT, "tokenized_emojis.txt"),
    os.path.join(DATA_ROOT, "tokenized_wikipedia.txt"),
    os.path.join(DATA_ROOT, "tokenized_dolly.txt"),
    os.path.join(DATA_ROOT, "tokenized_flan.txt"),
    os.path.join(DATA_ROOT, "tokenized_opus_en-hi.txt"),
    os.path.join(DATA_ROOT, "tokenized_opus_en-gu.txt"),
    os.path.join(DATA_ROOT, "tokenized_opus_en-ja.txt"),
    os.path.join(DATA_ROOT, "tokenized_opus_en-es.txt"),
    os.path.join(DATA_ROOT, "tokenized_opus_en-ko.txt"),
]


# ===== DEVICE SETUP =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# AMP scaler for GPU speed boost (30-50% faster)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))


# ===== CHECKPOINT SYSTEM =====
def save_checkpoint(model, optimizer, scheduler, epoch, global_step, avg_loss, step_in_epoch):
    
    checkpoint_data = {
        "model": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "step_in_epoch": step_in_epoch,
        "avg_loss": avg_loss,
    }
    temp = LATEST_PATH + ".tmp"
    torch.save(checkpoint_data, temp)
    os.replace(temp, LATEST_PATH)

    # save epoch-wise backup
    epoch_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt")
    temp_epoch = epoch_path + ".tmp"
    torch.save(checkpoint_data, temp_epoch)
    os.replace(temp_epoch, epoch_path)

    print(f"\n✅ [SAVED] epoch={epoch}, global_step={global_step}, step_in_epoch={step_in_epoch}")


def load_checkpoint():
    if not os.path.exists(LATEST_PATH):
        return None
    
    try:
        return torch.load(LATEST_PATH, map_location=DEVICE)
    except Exception as e:
        print(f"Checkpoint corrupted, starting fresh: {e}")
        return None


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

    # 🔥 DataParallel for multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Add optimizer and loss_fn after model init
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # ===== RESUME SUPPORT =====


    start_epoch = 0
    global_step = 0
    resume_step = 0
    ckpt = load_checkpoint()
    if ckpt:
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        resume_step = ckpt.get("step_in_epoch", 0)
        print(f"\n✅ RESUMED → epoch {start_epoch}, step_in_epoch {resume_step}")

    scheduler = None

    # ===== TRAINING LOOP =====
    model.train()
    print("\nStarting training...")

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nLoading dataset for epoch {epoch+1} (start_idx=0)")
        loader = get_dataloader(
            DATA_FILES,
            batch_size=BATCH_SIZE,
            shuffle=True,  # ✅ important
            start_idx=0
        )
        print(f"Total batches this epoch: {len(loader)}")

        if scheduler is None:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=(len(loader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
            )
            if ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        total_loss = 0.0

        progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}")

        for step, (input_ids, targets, attention_mask) in progress_bar:
            # ✅ ONLY skip for SAME epoch
            if epoch == start_epoch and step < resume_step:
                continue

            # Move to device
            input_ids = input_ids.to(DEVICE)
            targets = targets.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(input_ids, attention_mask)
                vocab_size = logits.shape[-1]
                loss = loss_fn(
                    logits.contiguous().view(-1, vocab_size),
                    targets.contiguous().view(-1)
                )

            if torch.isnan(loss):
                print("NaN detected! Skipping batch...")
                continue

            loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRADIENT)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            progress_bar.set_postfix({
                "loss": f"{total_loss / (step + 1):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                "step": global_step
            })

            if (step + 1) % (200 * GRADIENT_ACCUMULATION_STEPS) == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, total_loss / (step + 1), step)

        # Save at end of epoch — next epoch starts from step 0
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch + 1,
            global_step,
            total_loss / len(loader),
            0
        )


if __name__ == "__main__":
    train()