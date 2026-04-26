"""
Integration test: Connect Embedding Layer directly to existing DataLoader
Tests end-to-end flow with REAL tokenized conversational data from your dataset
"""
import torch
from dataloader import get_dataloader
from embedding import ConversationalEmbedding


def test_full_pipeline():
    print("=== Running Embedding + DataLoader Integration Test ===")
    
    # 1. Load your actual dataset (OpenAssistant conversational data)
    print("\n1. Loading real dataset...")
    data_files = [
        "tokenized_OpenAssistant_oasst1.txt"
    ]
    
    loader = get_dataloader(data_files, batch_size=8, shuffle=True)
    print(f"   Loaded {len(loader.dataset)} real sequences")
    
    # 2. Initialize embedding layer
    print("\n2. Initializing embedding layer...")
    embedding = ConversationalEmbedding()
    
    # 3. Test with multiple real batches
    print("\n3. Testing with real data batches:")
    
    total_tested = 0
    for batch_idx, (inputs, targets, attention_mask) in enumerate(loader):
        # Forward pass through embedding layer
        output = embedding(inputs)
        
        # Verify shapes
        batch_size, seq_len = inputs.shape
        expected_shape = (batch_size, seq_len, 384)
        
        assert output.shape == expected_shape, f"Shape mismatch at batch {batch_idx}: {output.shape} != {expected_shape}"
        
        # Verify no NaN values
        assert not torch.isnan(output).any(), f"NaN values found in batch {batch_idx}"
        
        # Verify padding positions are zeroed
        pad_mask = inputs == embedding.pad_token_id
        if pad_mask.any():
            pad_embeddings = output[pad_mask]
            assert torch.allclose(pad_embeddings, torch.zeros_like(pad_embeddings), atol=1e-6), "Padding positions not properly masked"
        
        print(f"   Batch {batch_idx}: OK - shape={output.shape}, mean={output.mean():.4f}, std={output.std():.4f}")
        
        total_tested += batch_size
        
        if batch_idx >= 5:
            break
    
    print(f"\n✅ Integration test PASSED!")
    print(f"✅ Successfully processed {total_tested} real conversational sequences")
    print(f"✅ Embedding layer is fully compatible with your existing dataloader")
    print(f"\n✅ Ready for Transformer blocks next!")


if __name__ == "__main__":
    test_full_pipeline()