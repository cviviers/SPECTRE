from typing import List, Callable, Optional

import torch
from monai.data import list_data_collate


def extended_collate_dino(samples_list: List) -> dict:
    """
    Applies MONAI's list_data_collate first and then extends it with DINOv2 masking logic.

    Args:
        samples_list: List of samples containing 'global_crops' and 'local_crops'.
        mask_ratio: Tuple defining the range of masking ratios.
        mask_probability: Probability of applying masking.
        dtype: Data type to cast the collated tensors.
        n_tokens: Number of tokens for masking.
        mask_generator: Function to generate masks.

    Returns:
        A dictionary with collated global/local crops and corresponding masks.
    """
    # Apply MONAI's list_data_collate
    collated_data = list_data_collate(samples_list)

    # Extract crops
    global_views = torch.cat(collated_data["image_global_views"], dim=0)
    local_views = torch.cat(collated_data["image_local_views"], dim=0)

    return {
        "global_views": global_views,
        "local_views": local_views,
    }
    

def extended_collate_siglip(
    samples_list: List,
    tokenizer: Optional[Callable] = None,
) -> dict:
    """
    Applies SigLIP collate and then extends it with tokenization logic.
    
    Args:
        samples_list: List of samples containing 'image' and 'report'.
        tokenizer: Tokenizer function to apply on the reports.
    
    Returns:
        A dictionary with collated images and tokenized text.
    """
    # [B][N][C, H, W, D] --> [B, N, C, H, W, D]
    collated_data = dict()
    collated_data["image"] = torch.stack([
        torch.stack([
            s["image"] for s in sample
        ], dim=0) for sample in samples_list
    ], dim=0)
    collated_data["report"] = [sample[0]["report"] for sample in samples_list]

    tokenizer_output = tokenizer.batch_encode_plus(
        collated_data["report"], 
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=1024,
    )
    
    collated_data["input_ids"] = torch.tensor(tokenizer_output["input_ids"])
    collated_data["attention_mask"] = torch.tensor(tokenizer_output["attention_mask"])

    return collated_data
