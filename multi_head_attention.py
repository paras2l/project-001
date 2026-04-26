"""
Multi-Head Causal Masked Self-Attention Block
Fully optimized and aligned with existing project pipeline
"""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, dropout=0.1):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Combined QKV projection (3x faster than separate layers)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim] - direct output from Embedding layer
            attention_mask: [batch_size, seq_len] - from DataLoader (1 = valid, 0 = padding)
        
        Returns:
            out: [batch_size, seq_len, embed_dim] - same shape as input
        """
        B, T, C = x.shape
        
        # Single projection for Q, K, V combined
        qkv = self.qkv_proj(x)  # [B, T, 3 * embed_dim]
        
        # Split and reshape into heads
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, head_dim]
        
        Q, K, V = qkv.unbind(0)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, heads, T, T]
        scores = scores / math.sqrt(self.head_dim)
        
        # Causal mask: prevent looking at future tokens
        causal_mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T] explicit broadcast shape
        
        # Combine with padding mask if provided
        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [B, 1, 1, T]
            combined_mask = causal_mask & pad_mask
        else:
            combined_mask = causal_mask
        
        scores = scores.masked_fill(~combined_mask, -1e9)
        
        # Numerical stability
        scores = scores - scores.max(dim=-1, keepdim=True).values
        
        # Softmax + dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [B, heads, T, head_dim]
        
        # Merge heads back together
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, T, C)
        
        # Final projection
        out = self.out_proj(out)
        
        return out


if __name__ == "__main__":
    print("Testing Multi-Head Attention layer...")
    
    # Exact parameters for our project
    attn = MultiHeadAttention(embed_dim=384, num_heads=6)
    
    # Test shape matching DataLoader outputs
    dummy_x = torch.randn(4, 256, 384)  # batch=4, seq_len=256, embed_dim=384
    dummy_mask = torch.randint(0, 2, (4, 256))
    
    output = attn(dummy_x, dummy_mask)
    
    print(f"OK Input shape:  {dummy_x.shape}")
    print(f"OK Output shape: {output.shape}")
    print(f"OK Mean value:   {output.mean():.4f}")
    print(f"OK Std value:    {output.std():.4f}")
    
    assert output.shape == dummy_x.shape, f"Shape mismatch! {output.shape} != {dummy_x.shape}"
    assert not torch.isnan(output).any(), "NaN values detected"
    
    print("\nOK Multi-Head Attention layer test passed!")