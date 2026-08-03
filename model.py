import torch
import torch.nn as nn
from transformers import AutoModel
from config import ContentSafetyConfig

class MultimodalEmbeddingBridge(nn.Module):
    """
    Production Multimodal Projection Architecture.
    Maps text and vision into a shared, normalized space to enable 
    dynamic similarity profiling and traditional inference capabilities.
    """
    def __init__(self, config: ContentSafetyConfig):
        super().__init__()
        self.config = config
        
        # Ingest foundational transformer backbones
        self.text_backbone = AutoModel.from_pretrained(config.TEXT_MODEL)
        self.vision_backbone = AutoModel.from_pretrained(config.VISION_MODEL).vision_model
        
        # Structural projection heads (768 -> 256 dimensions)
        self.text_projection = nn.Linear(768, config.PROJECTION_DIM)
        self.vision_projection = nn.Linear(768, config.PROJECTION_DIM)
        
        # Unified Latent Embedding space transformation
        self.unified_space = nn.Linear(config.PROJECTION_DIM * 2, config.PROJECTION_DIM)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        # Extract and project text vectors
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state[:, 0, :]  
        text_features = self.text_projection(text_embeds)       
        
        # Extract and project visual tensor features
        vision_outputs = self.vision_backbone(pixel_values=pixel_values)
        vision_embeds = vision_outputs.pooler_output            
        vision_features = self.vision_projection(vision_embeds) 
        
        # Execute Late Fusion Concatenation
        fused = torch.cat((text_features, vision_features), dim=1)
        
        # Project into metric space and L2-normalize
        latent_vectors = self.unified_space(fused)
        return nn.functional.normalize(latent_vectors, p=2, dim=1)
