"""
Full Causal Transformer Language Model
Complete end-to-end architecture for conversational LLM
100% aligned with existing project pipeline
"""
import torch
import torch.nn as nn

from embedding import ConversationalEmbedding
from transformer_block import TransformerBlock


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size=50262,
        embed_dim=384,
        num_heads=6,
        num_layers=6,
        dropout=0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Embedding layer (existing implementation)
        self.embedding = ConversationalEmbedding()
        
        # Stack of transformer decoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Language modeling output head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying (reduces parameters + improves training)
        self.lm_head.weight = self.embedding.token_embedding.weight

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [batch_size, seq_len] - direct output from DataLoader
            attention_mask: [batch_size, seq_len] - direct output from DataLoader
        
        Returns:
            logits: [batch_size, seq_len, vocab_size] - token prediction logits
        """
        
        # Convert token ids to embeddings
        x = self.embedding(input_ids)
        
        # Pass through each transformer block
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final normalization before output
        x = self.ln_f(x)
        
        # Generate output logits
        logits = self.lm_head(x)
        
        return logits


if __name__ == "__main__":
    print("Testing Full Transformer Model...")
    
    # Exact parameters matching our project
    model = TransformerModel(
        vocab_size=50262,
        embed_dim=384,
        num_heads=6,
        num_layers=8
    )
    
    # Test shape matching DataLoader outputs
    dummy_input = torch.randint(0, 50262, (4, 256))  # batch=4, seq_len=256
    dummy_mask = torch.randint(0, 2, (4, 256))
    
    logits = model(dummy_input, dummy_mask)
    
    print(f"OK Input shape:  {dummy_input.shape}")
    print(f"OK Logits shape: {logits.shape}")
    print(f"OK Mean value:   {logits.mean():.4f}")
    print(f"OK Std value:    {logits.std():.4f}")
    
    assert logits.shape == (4, 256, 50262), f"Shape mismatch! {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN values detected"
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nOK Total model parameters: {total_params:,}")
    print("\nOK Full Transformer Model test passed!")
