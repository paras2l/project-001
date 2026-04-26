"""
Conversational LLM Embedding Layer
Optimized for existing project pipeline with exact matched parameters
"""
import torch
import torch.nn as nn
import math


class ConversationalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        
        # EXACT VALUES MATCHING YOUR EXISTING TOKENIZER + DATALOADER
        self.vocab_size = 50268      # 50257 GPT2 + 11 special tokens
        self.embed_dim = 384         # Optimal for 1M sequence dataset
        self.max_seq_len = 1024      # Matches dataloader.py MAX_SEQ_LEN
        self.pad_token_id = 50257    # Matches your PAD token ID
        
        # 1. Token Embeddings with padding mask and proper initialization
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=self.pad_token_id
        )
        
        # 2. Fixed Sinusoidal Positional Embedding (stable for generalization)
        self._create_sinusoidal_embeddings()
        
        # 3. Regularization Dropout
        self.dropout = nn.Dropout(0.1)
        
        # 4. Embedding scaling factor (fixes gradient magnitude issue)
        self.scale = math.sqrt(self.embed_dim)
        
        # Initialize weights properly
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def _create_sinusoidal_embeddings(self):
        """
        Fixed sinusoidal positional embeddings
        Stable training, good sequence generalization
        """
        position = torch.arange(self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * (-math.log(10000.0) / self.embed_dim))
        
        pe = torch.zeros(self.max_seq_len, self.embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trainable)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe_buffer', pe)

    def forward(self, input_ids):
        """
        Args:
            input_ids: [batch_size, seq_len] - DIRECT OUTPUT FROM YOUR DATALOADER
        
        Returns:
            embeddings: [batch_size, seq_len, embed_dim] ready for transformer input
        """
        seq_len = input_ids.shape[1]
        
        # Get token embeddings with proper scaling
        token_emb = self.token_embedding(input_ids) * self.scale
        
        # Add positional embeddings
        embeddings = token_emb + self.pe_buffer[:, :seq_len, :]
        
        # Zero out padding positions (critical fix)
        pad_mask = (input_ids == self.pad_token_id).unsqueeze(-1)
        embeddings = embeddings * (~pad_mask)
        
        # Apply regularization
        embeddings = self.dropout(embeddings)
        
        return embeddings


if __name__ == "__main__":
    # Quick self test
    print("Testing Embedding Layer...")
    model = ConversationalEmbedding()
    
    # Test with shape matching your dataloader batches
    dummy_batch = torch.randint(0, 50262, (8, 256))  # batch=8, seq_len=256
    output = model(dummy_batch)
    
    print(f"OK Input shape:  {dummy_batch.shape}")
    print(f"OK Output shape: {output.shape}")
    print(f"OK Mean value:   {output.mean():.4f}")
    print(f"OK Std value:    {output.std():.4f}")
    
    assert output.shape == (8, 256, 384), f"Wrong output shape {output.shape}"
    print("\nOK Embedding layer test passed!")