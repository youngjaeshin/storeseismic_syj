# src/models/__init__.py

from .bert_model_setup import create_bert_config, get_bert_model
from .custom_heads import (
    BertOnlyMLMHead,
    DenoisingHead,
    VelpredHead,
    Velpred2DHead,
    FaultpredHead,
    FaultsignHead,
    # Add other heads if you want to expose them directly
    # FirstArrivalHead,
    # FirstBreakHead3,
)

__all__ = [
    "create_bert_config",
    "get_bert_model",
    "BertOnlyMLMHead",
    "DenoisingHead",
    "VelpredHead",
    "Velpred2DHead",
    "FaultpredHead",
    "FaultsignHead",
]