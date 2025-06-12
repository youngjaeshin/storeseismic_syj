# src/models/custom_heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper Modules (from original storseismic/modules.py)
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return 0

# --- Prediction Heads ---

class BertOnlyMLMHead(nn.Module):
    """Head for Masked Language Model prediction task (pre-training)."""
    def __init__(self, config):
        super().__init__()
        # config.vocab_size here refers to the feature dimension of one token
        # (e.g., number of time samples in a trace, or number of receivers in a time-slice)
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
        # self.predictions.decoder = Identity() # Original had this, implies no separate decoder weights

    def forward(self, sequence_output): # sequence_output shape: (batch_size, seq_len, hidden_size)
        output = self.predictions(sequence_output) # Output shape: (batch_size, seq_len, vocab_size/feature_dim)
        return output

class DenoisingHead(nn.Module):
    """Head for Denoising task (similar to MLM head)."""
    def __init__(self, config):
        super().__init__()
        # config.vocab_size should match the feature dimension of the token to be reconstructed
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        return output

class VelpredHead(nn.Module):
    """Head for 1D Velocity Profile Prediction."""
    def __init__(self, config):
        super().__init__()
        # config.vel_size is the target output dimension for the 1D velocity profile
        self.predictions = nn.Linear(config.hidden_size, config.vel_size)
        
        # vel_min and vel_max should be provided in the config, e.g., config.model.vel_min
        # These might be loaded from dataset statistics during training setup and passed to config.
        self.vel_min = config.model.get('vel_min', 1500.0) # Default or from config
        self.vel_max = config.model.get('vel_max', 4500.0) # Default or from config

    def forward(self, sequence_output): # sequence_output shape: (batch_size, seq_len, hidden_size)
        # Original script took mean over the first token's output: output = torch.mean(output[:, :1, :], dim=1)
        # This implies using the [CLS] token or first sequence element for prediction.
        # Let's assume the relevant representation is already pooled or is the first token's.
        # If sequence_output is [CLS] representation: (batch_size, hidden_size)
        # If sequence_output is all tokens: (batch_size, seq_len, hidden_size), then pool it.
        if sequence_output.ndim == 3: # If full sequence output
             pooled_output = torch.mean(sequence_output[:, 0, :], dim=0, keepdim=True) # Use first token [CLS] like
             # Or simply sequence_output[:, 0, :] if batch processing is handled upstream.
             # The original VelpredHead took (batch, seq_len, hidden_size) and did output[:, :1, :].mean(dim=1)
             # this means it uses the first token from seq_len for prediction, then outputs (batch_size, vel_size)
             pooled_output = sequence_output[:, 0, :] # (batch_size, hidden_size)

        else: # Assuming (batch_size, hidden_size)
            pooled_output = sequence_output

        output = self.predictions(pooled_output) # (batch_size, vel_size)
        
        # Rescale output from approx [-1, 1] (if model output is tanh-like) to [vel_min, vel_max]
        # The original script did: self.vel_min + (output + 1) * (self.vel_max - self.vel_min) * 0.5
        # This assumes 'output' from linear layer is then somehow normalized to [-1, 1]
        # Or if model output naturally falls in a range that (x+1)/2 maps to [0,1]
        # A common practice is to apply tanh to the linear layer output if [-1,1] is desired before scaling.
        output = torch.tanh(output) # Explicitly ensure range for scaling
        output = self.vel_min + (output + 1.0) * 0.5 * (self.vel_max - self.vel_min)
        return output

class Velpred2DHead(nn.Module):
    """Head for 2D Velocity Field Prediction."""
    def __init__(self, config):
        super().__init__()
        # These should come from config, e.g., config.model.nx_vel_physical, config.model.nz_vel_physical
        self.nx = config.model.get('nx_velocity_field', 128) # Example default
        self.nz = config.model.get('nz_velocity_field', 128) # Example default
        self.vel_min = config.model.get('vel_min', 1500.0)
        self.vel_max = config.model.get('vel_max', 4500.0)
        
        self.predictions = nn.Linear(config.hidden_size, self.nx * self.nz)

    def forward(self, sequence_output): # (batch, seq_len, hidden_size)
        # Strategy: mean pool over sequence length, then predict, reshape, scale
        if sequence_output.ndim == 3:
            pooled_output = torch.mean(sequence_output, dim=1) # (batch, hidden_size)
        else: # Assuming (batch_size, hidden_size)
            pooled_output = sequence_output
            
        out_flat = self.predictions(pooled_output) # (batch, nx * nz)
        out_reshaped = out_flat.view(-1, self.nx, self.nz) # (batch, nx, nz)
        
        out_scaled = torch.tanh(out_reshaped) # Ensure in [-1, 1]
        out_scaled = self.vel_min + (out_scaled + 1.0) * 0.5 * (self.vel_max - self.vel_min)
        return out_scaled

class FaultpredHead(nn.Module):
    """Head for Binary Fault Prediction (presence/absence)."""
    def __init__(self, config):
        super().__init__()
        # Predicts a single logit for binary classification
        self.predictions = nn.Linear(config.hidden_size, 1)

    def forward(self, sequence_output): # (batch, seq_len, hidden_size)
        # Typically uses the [CLS] token's representation (first token)
        # Original script: outputs.logits[:, 0, :1]
        if sequence_output.ndim == 3:
            cls_representation = sequence_output[:, 0, :] # (batch, hidden_size)
        else: # Assuming (batch_size, hidden_size)
            cls_representation = sequence_output
        
        output_logits = self.predictions(cls_representation) # (batch, 1)
        return output_logits

class FaultsignHead(nn.Module):
    """Head for Fault Type/Sign Classification (multi-class)."""
    def __init__(self, config):
        super().__init__()
        # num_fault_types should be in config, e.g., config.model.num_fault_types (e.g., 3)
        num_classes = config.model.get('num_fault_types', 3)
        
        # Original had a few layers, let's make it configurable or keep simple
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size // 2 if config.hidden_size // 2 > num_classes else num_classes*2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_size // 2 if config.hidden_size // 2 > num_classes else num_classes*2, num_classes)
        # Simpler version: self.predictions = nn.Linear(config.hidden_size, num_classes)


    def forward(self, sequence_output): # (batch, seq_len, hidden_size)
        if sequence_output.ndim == 3:
            cls_representation = sequence_output[:, 0, :] # (batch, hidden_size)
        else:
            cls_representation = sequence_output
        
        x = self.relu(self.fc1(cls_representation))
        output_logits = self.fc2(x) # (batch, num_classes)
        # output_logits = self.predictions(cls_representation) # For simpler version
        return output_logits

# --- Other Heads from storseismic/modules.py (FirstArrival, FirstBreak, LowFreq) ---
# These might need config parameters like config.vocab_size if they predict over features
# or config.sequence_length if they predict for each token in sequence.

class FirstArrivalHead(nn.Module): # Predicts one value per sequence element
    def __init__(self, config):
        super(FirstArrivalHead, self).__init__()
        self.predictions = nn.Linear(config.hidden_size, 1)

    def forward(self, sequence_output): # (batch, seq_len, hidden_size)
        output = self.predictions(sequence_output) # (batch, seq_len, 1)
        return output.squeeze(-1) # (batch, seq_len)

class FirstBreakHead3(nn.Module): # Predicts over vocab_size (features) for each token
    def __init__(self, config):
        super().__init__()
        self.act_fn = nn.Sigmoid()
        # config.vocab_size here is likely the feature dimension of the input tokens if predicting at that resolution
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, sequence_output): # (batch, seq_len, hidden_size)
        # Original: output = self.act_fn(sequence_output) -> predictions(output) -> swapaxes
        # This means activation on hidden_state, then linear, then swap.
        activated_sequence_output = self.act_fn(sequence_output)
        output_logits = self.predictions(activated_sequence_output) # (batch, seq_len, vocab_size)
        output = output_logits.permute(0, 2, 1) # (batch, vocab_size, seq_len)
        return output

# Add other heads (FirstArrivalHead2, LowFreqHead, FirstBreakHead, FirstBreakHead2, FirstBreakHead4)
# similarly, paying attention to their output dimensions and how they use config.vocab_size.
# For example:
class LowFreqHead(nn.Module): # Assumes predicting features similar to MLM/Denoising
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        return output