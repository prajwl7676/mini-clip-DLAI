
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm


# Encode full dataset 

@torch.no_grad()
def encode_dataset(model, loader, device) -> dict:
    
    model.eval()

    all_image_embs = []
    all_text_embs  = []

    for images, tokens in tqdm(loader, desc='  encoding', leave=False, unit='batch'):
        images    = images.to(device, non_blocking=True)
        input_ids = tokens['input_ids'].to(device, non_blocking=True)
        attn_mask = tokens['attention_mask'].to(device, non_blocking=True)

        img_emb = model.encode_image(images)
        txt_emb = model.encode_text(input_ids, attn_mask)

        all_image_embs.append(img_emb.cpu())
        all_text_embs.append(txt_emb.cpu())

    return {
        'image_embeddings': torch.cat(all_image_embs, dim=0),  # (N, D)
        'text_embeddings':  torch.cat(all_text_embs,  dim=0),  # (N, D)
    }


#  Recall@K 

def recall_at_k(
    image_embeddings: torch.Tensor,
    text_embeddings:  torch.Tensor,
    ks: list = [1, 5, 10],
) -> dict:
    
    N = image_embeddings.shape[0]

    # Full N×N cosine similarity matrix
    # S[i, j] = similarity between text_i and image_j
    # Since both are L2-normalised: cosine sim = dot product
    sim_t2i = text_embeddings @ image_embeddings.T   # (N, N)
    sim_i2t = image_embeddings @ text_embeddings.T   # (N, N)

    # Ground truth: item i is matched with item i (diagonal)
    # labels[i] = i
    labels = torch.arange(N)

    results = {}

    for direction, sim in [('t2i', sim_t2i), ('i2t', sim_i2t)]:
        # Sort each row descending — highest similarity first
        # ranked[i] = indices of all N items sorted by similarity to query i
        ranked = sim.argsort(dim=1, descending=True)   # (N, N)

        for k in ks:
            # top_k_preds[i] = top-K retrieved indices for query i
            top_k = ranked[:, :k]                      # (N, k)

            # For query i, the correct answer is i
            # Check if i appears anywhere in top_k[i]
            correct = (top_k == labels.unsqueeze(1))   # (N, k) bool
            recall  = correct.any(dim=1).float().mean().item() * 100.0

            results[f'{direction}_R@{k}'] = round(recall, 2)

    return results


# Pretty-print results 

def print_results(results: dict, title: str = 'Retrieval Results') -> None:
    print(f'\n{title}')
    print('─' * 42)
    print(f'  {"Metric":<18} {"Score":>8}')
    print('─' * 42)
    for direction, label in [('t2i', 'Text → Image'), ('i2t', 'Image → Text')]:
        print(f'  {label}')
        for k in [1, 5, 10]:
            key   = f'{direction}_R@{k}'
            score = results.get(key, 'N/A')
            bar   = '█' * int(score / 5) if isinstance(score, float) else ''
            print(f'    R@{k:<4}  {score:>6.2f}%  {bar}')
    print('─' * 42)


# Qualitative retrieval examples 
@torch.no_grad()
def retrieval_examples(
    model,
    test_hf_dataset,
    tokenizer,
    get_transform_fn,
    device,
    query_indices: list = [0, 1, 2, 3],
    top_k: int = 5,
    save_path: str = None,
) -> None:
    
    from src.dataset import FlickrDataset
    from torch.utils.data import DataLoader

    model.eval()

    # Build a small dataset from the test split
    transform = get_transform_fn(train=False)
    test_ds   = FlickrDataset(test_hf_dataset, tokenizer, transform)
    loader    = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    # Encode the full test set
    embeddings = encode_dataset(model, loader, device)
    img_embs   = embeddings['image_embeddings']   
    txt_embs   = embeddings['text_embeddings']   

    n_queries = len(query_indices)
    fig = plt.figure(figsize=(3 * (top_k + 1), 3.5 * n_queries))
    gs  = gridspec.GridSpec(n_queries, top_k + 1, figure=fig,
                            hspace=0.4, wspace=0.1)

    for row, q_idx in enumerate(query_indices):
        # Query caption 
        query_caption = test_hf_dataset[q_idx]['caption'][0]

        # Similarity of this query text against all images
        query_emb = txt_embs[q_idx].unsqueeze(0)          # (1, D)
        sims      = (query_emb @ img_embs.T).squeeze(0)   # (N,)
        top_k_idx = sims.argsort(descending=True)[:top_k].tolist()

        # Query text panel 
        ax_txt = fig.add_subplot(gs[row, 0])
        ax_txt.axis('off')
        ax_txt.text(
            0.5, 0.5,
            f'Query {row+1}:\n\n"{query_caption}"',
            ha='center', va='center',
            fontsize=8, wrap=True,
            transform=ax_txt.transAxes,
        )
        ax_txt.set_facecolor('#F5F5F5')

        # Retrieved images
        for col, img_idx in enumerate(top_k_idx):
            ax = fig.add_subplot(gs[row, col + 1])
            img = test_hf_dataset[img_idx]['image'].convert('RGB')
            ax.imshow(img)
            ax.axis('off')

            rank  = col + 1
            is_correct = (img_idx == q_idx)

            # Green border for correct match, red for incorrect
            border_color = '#1D9E75' if is_correct else '#E05050'
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)

            label = f'#{rank} ✓' if is_correct else f'#{rank}'
            ax.set_title(label, fontsize=8,
                         color='#1D9E75' if is_correct else '#555')

    fig.suptitle(
        'Text → Image Retrieval — Top-5 results\n'
        '(green border = correct match)',
        fontsize=11, y=1.01,
    )

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'Saved to {save_path}')

    plt.show()
