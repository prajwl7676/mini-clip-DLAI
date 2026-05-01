# Mini CLIP — Image–Text Contrastive Learning
**DLAI 2025/2026 Project** · Sapienza University of Rome

A small-scale reproduction of [CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020) trained on a subset of Flickr30k. The model learns to align image and text embeddings using contrastive learning, and is evaluated on zero-shot image–text retrieval.

---
 
## What this project does
 
1. **Trains** a dual-encoder model (ResNet-50 + DistilBERT) with a symmetric InfoNCE loss on ~10k Flickr30k image-caption pairs
2. **Evaluates** zero-shot retrieval performance using Recall@1, R@5, R@10 in both directions (image→text and text→image)
3. **Ablates** the effect of batch size (32 / 64 / 128) on retrieval quality — the main experimental contribution
---

## Key references
 
- Radford et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). OpenAI / ICML 2021.
- Young et al. (2014). [From image descriptions to visual denotations](https://aclanthology.org/Q14-1006/). Flickr30k dataset.
- Oord et al. (2018). [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748). InfoNCE loss.
- OpenAI CLIP — original repo(https://github.com/openai/CLIP)
- HuggingFace CLIP — transformers library (https://huggingface.co/docs/transformers/model_doc/clip)

---