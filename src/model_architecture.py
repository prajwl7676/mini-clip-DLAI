import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from transformers import DistilBertModel


#  Image Encoder 

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256, freeze_layers: bool = True):
        super().__init__()

        resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.projection = nn.Sequential(
            nn.Linear(2048, embedding_dim, bias=False),
            nn.LayerNorm(embedding_dim),
        )

        if freeze_layers:
            self._freeze_early_layers(resnet)

    def _freeze_early_layers(self, resnet):
        layers_to_freeze = [
            resnet.conv1,
            resnet.bn1,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        ]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        features = self.backbone(images)       # (B, 2048, 1, 1)
        features = features.flatten(start_dim=1)  # (B, 2048)

        embeddings = self.projection(features) # (B, embedding_dim)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


# Text Encoder 

class TextEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        model_name: str = "distilbert-base-uncased",
    ):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(768, embedding_dim, bias=False),
            nn.LayerNorm(embedding_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]   # (B, 768)

        embeddings = self.projection(cls_output)           # (B, embedding_dim)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


# CLIP Model 

class CLIPModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        temperature_init: float = 0.07,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(embedding_dim=embedding_dim)
        self.text_encoder  = TextEncoder(embedding_dim=embedding_dim)
        self.log_temperature = nn.Parameter(
            torch.tensor(temperature_init).log()
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple:
       
        image_embeddings = self.image_encoder(images)
        text_embeddings  = self.text_encoder(input_ids, attention_mask)

        return image_embeddings, text_embeddings, self.temperature


#  Utility: count trainable parameters 

def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    frozen,
        "trainable_pct": round(100 * trainable / total, 1),
    }
