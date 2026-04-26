"""
Full Transformer Decoder Block
Pre-Norm architecture optimized for conversational language modeling
100% aligned with existing project pipeline
"""
import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, embed_dim=384, expansion_factor=4, dropout=0.1):
        super().__init__()
        
        hidden_dim = embed_dim * expansion_factor
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),                     # Standard activation for modern LLMs
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, dropout=0.1):
        super().__init__()
        
        # Multi-Head Causal Self Attention
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed Forward Network
        self.ff = FeedForward(
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Pre-Normalization layers (GPT style)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            x: [batch_size, seq_len, embed_dim] - same shape as input
        """
        # Attention sub-block with residual connection
        attn_out = self.attn(self.ln1(x), attention_mask)
        x = x + self.dropout(attn_out)
        
        # Feed Forward sub-block with residual connection
        ff_out = self.ff(self.ln2(x))
        x = x + self.dropout(ff_out)
        
        return x


if __name__ == "__main__":
    print("Testing Transformer Block...")
    
    # Exact parameters matching our project
    block = TransformerBlock(embed_dim=384, num_heads=6)
    
    # Test shape matching DataLoader outputs
    dummy_x = torch.randn(4, 256, 384)  # batch=4, seq_len=256, embed_dim=384
    dummy_mask = torch.randint(0, 2, (4, 256))
    
    output = block(dummy_x, dummy_mask)
    
    print(f"OK Input shape:  {dummy_x.shape}")
    print(f"OK Output shape: {output.shape}")
    print(f"OK Mean value:   {output.mean():.4f}")
    print(f"OK Std value:    {output.std():.4f}")
    
    assert output.shape == dummy_x.shape, f"Shape mismatch! {output.shape} != {dummy_x.shape}"
    assert not torch.isnan(output).any(), "NaN values detected"
    
    print("\nOK Transformer Block test passed!")