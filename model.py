import torch
import torch.nn as nn
from transformers import AutoModel
from config import ContentSafetyConfig

class LateFusionClassifier(nn.Module):
    """
    Multimodal network combining text and vision embeddings.
    Designed for high-throughput batch evaluation.
    """
    def __init__(self, config: ContentSafetyConfig):
        super().__init__()
        self.config = config
        
        # Load backbones (typically pinned to localized fine-tuned weights in production)
        self.text_backbone = AutoModel.from_pretrained(config.TEXT_MODEL)
        self.vision_backbone = AutoModel.from_pretrained(config.VISION_MODEL).vision_model
        
        # Projection Layers
        self.text_projection = nn.Linear(config.TEXT_EMBED_DIM, config.PROJECTION_DIM)
        self.vision_projection = nn.Linear(config.VISION_EMBED_DIM, config.PROJECTION_DIM)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(config.PROJECTION_DIM * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, config.NUM_CLASSES)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        # Extract features
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state[:, 0, :]  # CLS Token Pooling
        text_features = self.text_projection(text_embeds)
        
        vision_outputs = self.vision_backbone(pixel_values=pixel_values)
        vision_embeds = vision_outputs.pooler_output
        vision_features = self.vision_projection(vision_embeds)
        
        # Merge modalities
        fused = torch.cat((text_features, vision_features), dim=1)
        return self.classifier(fused)
