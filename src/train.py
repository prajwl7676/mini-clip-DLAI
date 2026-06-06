
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.loss import clip_loss


# Training epoch 

def train_epoch(
    model,
    loader,
    optimizer,
    device,
    max_grad_norm: float = 1.0,
) -> dict:
    
    model.train()   # enable dropout, batch norm in training mode

    total_loss   = 0.0
    num_batches  = 0

    # tqdm wraps the loader to show a live progress bar
    progress = tqdm(loader, desc='  train', leave=False, unit='batch')

    for images, tokens in progress:

        # Move data to GPU 
        images     = images.to(device, non_blocking=True)
        input_ids  = tokens['input_ids'].to(device, non_blocking=True)
        attn_mask  = tokens['attention_mask'].to(device, non_blocking=True)

        # Forward pass 
        img_emb, txt_emb, temperature = model(images, input_ids, attn_mask)

        # Compute loss 
        loss = clip_loss(img_emb, txt_emb, temperature)

        # Backward pass 
        optimizer.zero_grad()      # clear gradients from previous batch
        loss.backward()            # compute gradients via backprop

        # Gradient clipping 
        # Prevents any single gradient from becoming too large,
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Weight update 
        optimizer.step()

        # Accumulate metrics 
        total_loss  += loss.item()
        num_batches += 1

        # Update live progress bar with current loss and τ
        progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'τ':    f'{temperature.item():.4f}',
        })

    avg_loss = total_loss / num_batches

    return {
        'loss':        avg_loss,
        'temperature': temperature.item(),
    }


# Validation 

def validate(
    model,
    loader,
    device,
) -> dict:
   
    model.eval()    # disable dropout, use running stats for batch norm

    total_loss  = 0.0
    num_batches = 0

    progress = tqdm(loader, desc='    val', leave=False, unit='batch')

    with torch.no_grad():   # no gradient computation needed 
        for images, tokens in progress:

            images    = images.to(device, non_blocking=True)
            input_ids = tokens['input_ids'].to(device, non_blocking=True)
            attn_mask = tokens['attention_mask'].to(device, non_blocking=True)

            img_emb, txt_emb, temperature = model(images, input_ids, attn_mask)
            loss = clip_loss(img_emb, txt_emb, temperature)

            total_loss  += loss.item()
            num_batches += 1

            progress.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches

    return {
        'loss':        avg_loss,
        'temperature': temperature.item(),
    }


# Checkpoint helpers 

def save_checkpoint(
    model,
    optimizer,
    epoch:        int,
    train_loss:   float,
    val_loss:     float,
    loss_history: dict,
    save_path:    str,
) -> None:
   
    checkpoint = {
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'train_loss':      train_loss,
        'val_loss':        val_loss,
        'loss_history':    loss_history,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model,
    optimizer,
    load_path: str,
    device,
) -> dict:
    
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint
