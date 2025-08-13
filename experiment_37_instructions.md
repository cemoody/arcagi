# Experiment 37: Multi-File Training with Context-Aware Hypernetworks

## Overview
This experiment extends ex36 by introducing simultaneous training on multiple files with file-specific context modulation using hypernetwork-like bilinear attention mechanisms.

## Background
- **Base**: Copy implementation from ex36
- **Previous approach**: Single-file training (one file at a time, each containing train/test examples)
- **New approach**: Train on multiple files simultaneously with file-aware attention modulation

## Core Concept: Context-Aware Attention via Bilinear Maps

Instead of simply concatenating file information or adding it as an embedding, we'll use a more sophisticated approach where each file's context dynamically modulates the attention mechanism through bilinear transformations.

## Technical Implementation Details

### 1. File Context Representation
```python
# Each file gets a learnable context vector
context_embeddings = nn.Embedding(num_files, context_dim)
file_context = context_embeddings(file_id)  # Shape: [batch_size, context_dim]
```

### 2. Bilinear Attention Modulation

The key innovation is using bilinear maps to generate file-specific attention matrices:

```python
# For each attention matrix (Wq, Wk, Wv), we have a 3D tensor
# Shape: [d_model, d_model, context_dim]
bilinear_tensor_q = nn.Parameter(torch.randn(d_model, d_model, context_dim))
bilinear_tensor_k = nn.Parameter(torch.randn(d_model, d_model, context_dim))
bilinear_tensor_v = nn.Parameter(torch.randn(d_model, d_model, context_dim))

# Generate file-specific attention matrices
# Einstein notation: ijk,k->ij (tensor contraction along context dimension)
Wq_modulated = torch.einsum('ijk,bk->bij', bilinear_tensor_q, file_context)
Wk_modulated = torch.einsum('ijk,bk->bij', bilinear_tensor_k, file_context)
Wv_modulated = torch.einsum('ijk,bk->bij', bilinear_tensor_v, file_context)
```

### 3. Modified Attention Mechanism

```python
class ContextModulatedAttention(nn.Module):
    def __init__(self, d_model, context_dim):
        super().__init__()
        # Base attention matrices (optional, can be zero-initialized)
        self.W_q_base = nn.Linear(d_model, d_model, bias=False)
        self.W_k_base = nn.Linear(d_model, d_model, bias=False)
        self.W_v_base = nn.Linear(d_model, d_model, bias=False)
        
        # Bilinear tensors for modulation
        self.bilinear_q = nn.Parameter(torch.randn(d_model, d_model, context_dim))
        self.bilinear_k = nn.Parameter(torch.randn(d_model, d_model, context_dim))
        self.bilinear_v = nn.Parameter(torch.randn(d_model, d_model, context_dim))
        
    def forward(self, x, file_context):
        # x: [batch_size, seq_len, d_model]
        # file_context: [batch_size, context_dim]
        
        # Generate file-specific weight matrices
        Wq = self.W_q_base.weight + torch.einsum('ijk,bk->bij', self.bilinear_q, file_context)
        Wk = self.W_k_base.weight + torch.einsum('ijk,bk->bij', self.bilinear_k, file_context)
        Wv = self.W_v_base.weight + torch.einsum('ijk,bk->bij', self.bilinear_v, file_context)
        
        # Apply modulated transformations
        Q = torch.einsum('bsd,bde->bse', x, Wq)
        K = torch.einsum('bsd,bde->bse', x, Wk)
        V = torch.einsum('bsd,bde->bse', x, Wv)
        
        # Standard attention computation continues...
        return attention_output
```

## Implementation Steps

### Step 1: Copy ex36 to ex37
```bash
cp -r ex36/ ex37/
cd ex37/
```

### Step 2: Modify Data Loading
- Update data loader to handle multiple files simultaneously
- Add file ID tracking for each example
- Ensure batch contains examples from different files

```python
class MultiFileDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []
        self.file_ids = []
        
        for file_id, path in enumerate(file_paths):
            file_data = load_file(path)
            self.data.extend(file_data)
            self.file_ids.extend([file_id] * len(file_data))
    
    def __getitem__(self, idx):
        return {
            'input': self.data[idx]['input'],
            'target': self.data[idx]['target'],
            'file_id': self.file_ids[idx]
        }
```

### Step 3: Debug Configuration
For initial debugging on CPU with minimal resources:

```python
debug_config = {
    'num_files': 2,  # Start with just 2 files
    'd_model': 64,   # Tiny model dimension
    'n_heads': 2,    # Minimal attention heads
    'n_layers': 2,   # Shallow network
    'context_dim': 16,  # Small context dimension
    'batch_size': 4,
    'seq_len': 32,
    'training_steps': 1,  # Single step for debugging
    'device': 'cpu'
}
```

### Step 4: Training Loop Modifications

```python
def train_step(model, batch, optimizer):
    inputs = batch['input']
    targets = batch['target']
    file_ids = batch['file_id']
    
    # Get file contexts
    file_contexts = model.get_file_context(file_ids)
    
    # Forward pass with context
    outputs = model(inputs, file_contexts)
    
    # Compute loss
    loss = criterion(outputs, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

## Key Advantages

1. **Dynamic Adaptation**: Each file can effectively "fine-tune" the attention mechanism without changing the core model parameters
2. **Parameter Efficiency**: Shared base model with lightweight file-specific modulation
3. **Flexibility**: Can handle varying numbers of files without architectural changes
4. **Interpretability**: File contexts can be analyzed to understand file-specific adaptations

## Debugging Checklist

1. **Tensor Shapes**: Verify all bilinear operations produce correct shapes
2. **Gradient Flow**: Ensure gradients flow through bilinear tensors
3. **Memory Usage**: Monitor memory with multiple files (bilinear tensors can be large)
4. **Initialization**: Proper initialization of bilinear tensors (consider small values to start)
5. **File ID Tracking**: Verify correct file IDs are maintained throughout batching

## Expected Outcomes

- Model should show different attention patterns for different files
- Performance should match or exceed single-file training
- File contexts should become distinguishable in embedding space

## Next Steps After Debugging

1. Scale to more files
2. Experiment with context dimension sizes
3. Try different bilinear tensor initialization strategies
4. Consider regularization on bilinear tensors
5. Analyze learned file contexts for interpretability