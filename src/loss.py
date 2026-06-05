
import torch
import torch.nn.functional as F


def clip_loss(
    image_embeddings: torch.Tensor,
    text_embeddings:  torch.Tensor,
    temperature:      torch.Tensor,
) -> torch.Tensor:
    
    batch_size = image_embeddings.shape[0]

    #  Cosine similarity matrix
    # Both tensors are L2-normalised → dot product = cosine similarity
    # Shape: (B, B)
    # S[i, j] = similarity between image i and text j
    similarity_matrix = image_embeddings @ text_embeddings.T   # (B, B)

    #  Scale by temperature
    # temperature is a learnable scalar; dividing sharpens the distribution
    logits = similarity_matrix / temperature   # (B, B)

    # Targets and cross-entropy 
    # Ground-truth labels: image i matches caption i → label = i
    # torch.arange(B) = [0, 1, 2, ..., B-1]
    labels = torch.arange(batch_size, device=image_embeddings.device)

    # Image-side loss: for each row, the correct column is the diagonal
    # F.cross_entropy treats each row as a logit vector, labels[i]=i as target
    loss_images = F.cross_entropy(logits,   labels)   # rows   → correct column

    # Text-side loss: for each column, the correct row is the diagonal
    # logits.T flips rows and columns
    loss_texts  = F.cross_entropy(logits.T, labels)   # columns → correct row

    #  Symmetric average 
    loss = (loss_images + loss_texts) / 2.0

    return loss


def similarity_matrix(
    image_embeddings: torch.Tensor,
    text_embeddings:  torch.Tensor,
) -> torch.Tensor:
   
    return image_embeddings @ text_embeddings.T
