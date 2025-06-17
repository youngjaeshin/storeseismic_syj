# src/models/architecture.py
import torch
import torch.nn as nn
import math
import transformers
from transformers import BertConfig, BertForMaskedLM
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from typing import Optional

# --- START: BERT Core Components ---

class PositionalEncoding(nn.Module):
    """Fixed Sinusoidal Positional Encoding"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_enc_to_add = self.pe[:, :x.size(1), :]
        return self.dropout(pos_enc_to_add)

class CustomBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.position_embedding_type = getattr(config, "position_embedding_type", "sincos")

        if self.position_embedding_type == "sincos":
            self.position_embeddings = PositionalEncoding(
                d_model=config.hidden_size,
                max_len=config.max_position_embeddings,
                dropout=config.hidden_dropout_prob
            )
        elif self.position_embedding_type == "learned":
             self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.position_embeddings = None

        if getattr(config, "add_alibi", False):
            self.position_embeddings = None

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, inputs_embeds, **kwargs):
        embeddings = self.word_embeddings(inputs_embeds)
        if self.position_embeddings is not None:
            if isinstance(self.position_embeddings, PositionalEncoding):
                embeddings += self.position_embeddings(embeddings)
            elif isinstance(self.position_embeddings, nn.Embedding):
                seq_length = inputs_embeds.size(1)
                position_ids = self.position_ids[:, :seq_length]
                embeddings += self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# Pre-LN BERT Components
class PreLNBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class PreLNBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self = transformers.models.bert.modeling_bert.BertSelfAttention(config)
        self.output = PreLNBertSelfOutput(config)
        self.pruned_heads = set()
    
    def prune_heads(self, heads): # Standard prune_heads implementation
        if len(heads) == 0: return
        heads_to_prune, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads -= len(heads_to_prune)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads_to_prune)


    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False):
        normalized_hidden_states = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            normalized_hidden_states, attention_mask, head_mask,
            encoder_hidden_states, encoder_attention_mask,
            past_key_value, output_attentions
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class PreLNBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = transformers.activations.ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        normalized_hidden_states = self.LayerNorm(hidden_states)
        intermediate_output = self.dense(normalized_hidden_states)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        return intermediate_output

class PreLNBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

# --- END: BERT Core Components ---


# --- START: Prediction Heads ---

class BertOnlyMLMHead(nn.Module):
    """Head for Masked Language Model prediction task (pre-training)."""
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, sequence_output):
        return self.predictions(sequence_output)

class DenoisingHead(nn.Module):
    """Head for Denoising task (similar to MLM head)."""
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, sequence_output):
        return self.predictions(sequence_output)

class VelpredHead(nn.Module):
    """Head for 1D Velocity Profile Prediction."""
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vel_size)
        self.vel_min = getattr(config, 'vel_min', 1500.0)
        self.vel_max = getattr(config, 'vel_max', 4500.0)

    def forward(self, sequence_output):
        if sequence_output.ndim == 3:
            pooled_output = sequence_output[:, 0, :]
        else:
            pooled_output = sequence_output
        output = self.predictions(pooled_output)
        output = torch.tanh(output)
        output = self.vel_min + (output + 1.0) * 0.5 * (self.vel_max - self.vel_min)
        return output

class FaultpredHead(nn.Module):
    """Head for Binary Fault Prediction (presence/absence)."""
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, 1)

    def forward(self, sequence_output):
        if sequence_output.ndim == 3:
            cls_representation = sequence_output[:, 0, :]
        else:
            cls_representation = sequence_output
        return self.predictions(cls_representation)

# --- END: Prediction Heads ---


# --- START: Model Factory Functions ---

def create_bert_config(cfg_model_from_yaml, cfg_data_from_yaml):
    config = BertConfig()
    config.vocab_size = cfg_model_from_yaml['input_feature_dim']
    config.max_position_embeddings = cfg_model_from_yaml['sequence_length']
    config.hidden_size = cfg_model_from_yaml.get('hidden_size', 768)
    config.num_hidden_layers = cfg_model_from_yaml.get('num_hidden_layers', 12)
    config.num_attention_heads = cfg_model_from_yaml.get('num_attention_heads', 12)
    config.intermediate_size = config.hidden_size * cfg_model_from_yaml.get('intermediate_ffn_factor', 4)
    config.hidden_act = cfg_model_from_yaml.get('hidden_act', "gelu")
    config.hidden_dropout_prob = cfg_model_from_yaml.get('hidden_dropout_prob', 0.1)
    config.attention_probs_dropout_prob = cfg_model_from_yaml.get('attention_probs_dropout_prob', 0.1)
    config.layer_norm_eps = cfg_model_from_yaml.get('layer_norm_eps', 1e-12)
    config.position_embedding_type = cfg_model_from_yaml.get('position_embedding_type', "sincos")
    config.pre_ln = cfg_model_from_yaml.get('pre_ln', False)
    config.attention_type = cfg_model_from_yaml.get('attention_type', "default")
    config.add_alibi = cfg_model_from_yaml.get('add_alibi', False)
    config.output_attentions = cfg_model_from_yaml.get('output_attentions', True)
    config.output_hidden_states = cfg_model_from_yaml.get('output_hidden_states', True)
    return config

def _patch_transformers_modules(config_model):
    """Hugging Face 모듈을 사용자 정의 모듈로 교체합니다."""
    transformers.models.bert.modeling_bert.BertEmbeddings = CustomBertEmbeddings
    
    if config_model.pre_ln:
        print("Patching BERT with Pre-LN custom modules.")
        transformers.models.bert.modeling_bert.BertSelfOutput = PreLNBertSelfOutput
        transformers.models.bert.modeling_bert.BertAttention = PreLNBertAttention
        transformers.models.bert.modeling_bert.BertIntermediate = PreLNBertIntermediate
        transformers.models.bert.modeling_bert.BertOutput = PreLNBertOutput

def get_bert_model(model_config: BertConfig, checkpoint_path: Optional[str] = None):
    """
    사용자 정의 컴포넌트로 BERT 모델을 생성하고, 선택적으로 가중치를 로드합니다.
    """
    _patch_transformers_modules(model_config)
    
    print("Instantiating BertForMaskedLM with config...")
    model = BertForMaskedLM(config=model_config)

    if checkpoint_path:
        try:
            print(f"Loading model weights from checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model.load_state_dict(pretrained_dict, strict=False)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model checkpoint from {checkpoint_path}: {e}")
            print("Proceeding with a randomly initialized model.")
            
    return model

# --- END: Model Factory Functions ---