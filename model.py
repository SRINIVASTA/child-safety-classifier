import torch
import torch.nn as nn
from transformers import AutoModel
from config import ContentSafetyConfig

class LateFusionClassifier(nn.Module):
    """
    Multimodal network combining text and vision embeddings.
    Fixed matrix dimensions to prevent runtime projection errors.
    """
    def __init__(self, config: ContentSafetyConfig):
        super().__init__()
        self.config = config
        
        # Load backbones (typically pinned to localized fine-tuned weights in production)
        self.text_backbone = AutoModel.from_pretrained(config.TEXT_MODEL)
        self.vision_backbone = AutoModel.from_pretrained(config.VISION_MODEL).vision_model
        
        # Projection Layers: Mapping both raw modalities to a unified shared space
        self.text_projection = nn.Linear(config.TEXT_EMBED_DIM, config.PROJECTION_DIM)
        self.vision_projection = nn.Linear(config.VISION_EMBED_DIM, config.PROJECTION_DIM)
        
        # Classification Head: Handles concatenated projections
        # Projections are fused side-by-side, so input size must be PROJECTION_DIM * 2 (256 * 2 = 512)
        self.classifier = nn.Sequential(
            nn.Linear(config.PROJECTION_DIM * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, config.NUM_CLASSES)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        # 1. Extract and project Text features
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state[:, 0, :]  # CLS Token Pooling (Batch, 768)
        text_features = self.text_projection(text_embeds)       # Shape transitions to (Batch, 256)
        
        # 2. Extract and project Vision features
        vision_outputs = self.vision_backbone(pixel_values=pixel_values)
        vision_embeds = vision_outputs.pooler_output          # Shape is (Batch, 512)
        vision_features = self.vision_projection(vision_embeds) # Shape transitions to (Batch, 256)
        
        # 3. Late Fusion via Concatenation
        # Combines text (Batch, 256) and vision (Batch, 256) into a unified (Batch, 512) tensor
        fused = torch.cat((text_features, vision_features), dim=1)
        
        # 4. Final Classification Pass
        return self.classifier(fused)
