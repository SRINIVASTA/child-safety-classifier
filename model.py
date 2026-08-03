import torch
import torch.nn as nn
from transformers import AutoModel
from config import ContentSafetyConfig

class LateFusionClassifier(nn.Module):
    """
    Multimodal network combining text and vision embeddings.
    Structural solution designed to explicitly process 768 -> 256 projections.
    """
    def __init__(self, config: ContentSafetyConfig):
        super().__init__()
        self.config = config
        
        # 1. Initialize core backbones (Both output raw 768 dimensions)
        self.text_backbone = AutoModel.from_pretrained(config.TEXT_MODEL)
        self.vision_backbone = AutoModel.from_pretrained(config.VISION_MODEL).vision_model
        
        # 2. Projection Mappings (768 -> 256)
        # Fixes the dimension mismatch issue directly at layer initialization
        self.text_projection = nn.Linear(768, 256)
        self.vision_projection = nn.Linear(768, 256)
        
        # 3. Dense Classifier Head
        # Concatenated vectors form a clean 512 shape input layer (256 + 256)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.NUM_CLASSES)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        # Step A: Process Text (Batch, 768) -> Project to (Batch, 256)
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state[:, 0, :]  
        text_features = self.text_projection(text_embeds)       
        
        # Step B: Process Vision (Batch, 768) -> Project to (Batch, 256)
        vision_outputs = self.vision_backbone(pixel_values=pixel_values)
        vision_embeds = vision_outputs.pooler_output            
        vision_features = self.vision_projection(vision_embeds) 
        
        # Step C: Multimodal Fusion via Side-by-Side Concatenation (Batch, 512)
        fused = torch.cat((text_features, vision_features), dim=1)
        
        # Step D: Process through aligned classification layers
        return self.classifier(fused)
