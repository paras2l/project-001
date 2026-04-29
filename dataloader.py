"""
DataLoader for tokenized conversational datasets.
Loads tokenized .txt files and creates (input, target) batches for causal LM training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import random

# Configuration
PAD_TOKEN_ID = 50257  # <|pad|> added to GPT2 tokenizer
BOS_TOKEN_ID = 50258  # <|startoftext|>
EOS_TOKEN_ID = 50259  # <|endoftext|>
MAX_SEQ_LEN = 1024
BATCH_SIZE = 8

class TextDataset(Dataset):
    """
    Reads tokenized .txt files where each line is space-separated token IDs.
    Stores all sequences in memory for fast access.
    """
    def __init__(self, file_paths, max_length=MAX_SEQ_LEN, start_idx=0):
        self.sequences = []
        self.max_length = max_length
        
        for path in file_paths:
            if not os.path.exists(path):
                print(f"Warning: {path} not found, skipping.")
                continue
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = [int(x) for x in line.split()]
                    # Truncate if too long
                    if len(tokens) > max_length:
                        tokens = tokens[:max_length]
                    if len(tokens) > 0:
                        self.sequences.append(tokens)
        
        # Always load full dataset, never cut for resume
        print(f"Loaded TOTAL sequences: {len(self.sequences)}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Return a single token sequence."""
        return self.sequences[idx]


def collate_fn(batch, pad_token_id=PAD_TOKEN_ID):
    """
    Pads sequences to max length in batch and creates (input, target) pairs.
    Target = input shifted by 1 position (causal LM).
    """
    # Find max length in this batch
    max_len = max(len(seq) for seq in batch)
    
    input_batch = []
    target_batch = []
    attention_mask = []
    
    for seq in batch:
        # Pad sequence to same length
        padded = seq + [pad_token_id] * (max_len - len(seq))
        
        input_batch.append(padded)
        
        # Target is input shifted LEFT by 1 (same length as input)
        target = padded[1:] + [pad_token_id]
        target_batch.append(target)
        
        # Attention mask for transformer (1 = valid token, 0 = padding)
        attention_mask.append([1 if t != pad_token_id else 0 for t in padded])
    
    return (
        torch.tensor(input_batch, dtype=torch.long),
        torch.tensor(target_batch, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long)
    )


def get_dataloader(file_paths, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=None, start_idx=0):
    """
    Creates a DataLoader for tokenized text files.
    
    Args:
        file_paths: List of paths to tokenized .txt files
        batch_size: Number of sequences per batch
        shuffle: Whether to shuffle data each epoch
        num_workers: Number of worker processes (0 = main process only)
    
    Returns:
        DataLoader yielding (input_tensor, target_tensor) batches
    """
    dataset = TextDataset(file_paths, start_idx=start_idx)
    
    # Auto detect pin_memory based on CUDA availability
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return loader


if __name__ == "__main__":
    # Example usage with all your datasets
    data_files = [
        "tokenized_blended_skill_talk.txt",
        "tokenized_Cynaptics_persona-chat.txt",
        "tokenized_kaistlayner_empathy-dataset.txt",
        "tokenized_multiwoz.txt",
        "tokenized_OpenAssistant_oasst1.txt",
        "tokenized_ParlAI_blended_skill_talk.txt",
        "tokenized_tatsu-lab_alpaca.txt",
        "tokenized_cornell_movie_dialogs.txt",
        "tokenized_emojis.txt",
    ]
    
    print("Creating DataLoader...")
    loader = get_dataloader(data_files, batch_size=4)
    
    # Test: get one batch
    for batch_idx, (inputs, targets, mask) in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Input shape:  {inputs.shape}")   # [batch_size, seq_len]
        print(f"  Target shape: {targets.shape}")  # [batch_size, seq_len]
        print(f"  Mask shape:   {mask.shape}")     # [batch_size, seq_len]
        print(f"  Input sample:  {inputs[0][:10].tolist()}")
        print(f"  Target sample: {targets[0][:10].tolist()}")
        print(f"  Mask sample:  {mask[0][:10].tolist()}")
        
        # Verify shapes match exactly (critical fix)
        assert inputs.shape == targets.shape, f"Shape mismatch! Input {inputs.shape} != Target {targets.shape}"
        assert inputs.shape == mask.shape, f"Shape mismatch! Input {inputs.shape} != Mask {mask.shape}"
        
        if batch_idx >= 2:
            break
    
    print("\nDataLoader ready for training!")
