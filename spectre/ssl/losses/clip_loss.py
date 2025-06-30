import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(
        self, 
        normalize: bool = True,
        t: float = 0.07,
    ):
        """
        CLIP loss for aligning image and text embeddings.

        Args:
            normalize (bool): If True, embeddings are L2-normalized before computing logits.
            t (float): Temperature parameter for scaling the logits.
        """
        super().__init__()
        self.normalize = normalize
        self.t = nn.Parameter(torch.ones([]) * (1 / t))

    def forward(
        self, 
        zimg: torch.Tensor, 
        ztxt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the SigLIP loss between image and text embeddings.

        Args:
            zimg (torch.Tensor): Image embeddings of shape (batch_size, embedding_dim).
            ztxt (torch.Tensor): Text embeddings of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Optionally normalize the embeddings
        if self.normalize:
            zimg = F.normalize(zimg, dim=-1)
            ztxt = F.normalize(ztxt, dim=-1)

        # Compute the logits using outer product
        logits = torch.matmul(zimg, ztxt.T) * self.t

        # Ground truth labels are diagonal (i.e., perfect alignment)
        labels = torch.arange(zimg.size(0), device=zimg.device)

        # Cross entropy in both directions (image->text and text->image)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        # Average the two directions
        loss = (loss_i2t + loss_t2i) / 2.0

        return loss
