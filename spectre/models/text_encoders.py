import torch
import torch.nn as nn
import torch.nn.functional as F

from spectre.models.layers.patch_embed import RotaryEmbedding
from transformers import AutoModel, AutoTokenizer

class CLIPTextEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int = 49152,
                 embed_dim: int = 512,
                 context_length: int = 76,
                 transformer_width: int = 512,
                 transformer_heads: int = 8,
                 transformer_layers: int = 12):
        """
        Initializes the CLIP text encoder.
        
        Args:
            vocab_size: Size of the vocabulary.
            embed_dim: Dimension of the final text embedding.
            context_length: Maximum token length of the input text.
            transformer_width: Hidden dimension for transformer layers.
            transformer_heads: Number of attention heads.
            transformer_layers: Number of transformer encoder layers.
        """
        super().__init__()
        self.context_length = context_length

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        nn.init.normal_(self.positional_embedding, std=0.01)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_width, 
            nhead=transformer_heads, 
            dim_feedforward=transformer_width * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Final layer norm and projection to match image encoder dimensions
        self.ln_final = nn.LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the text encoder.
        
        Args:
            tokens: Tensor of shape (batch_size, context_length) containing token indices.
            
        Returns:
            Tensor of shape (batch_size, embed_dim) containing the text embeddings.
        """
        # Embed tokens and add positional information.
        x = self.token_embedding(tokens)  # (batch_size, context_length, transformer_width)
        x = x + self.positional_embedding  # broadcasting positional embeddings

        # Rearrange for transformer which expects (sequence_length, batch_size, model_dim)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # back to (batch_size, context_length, transformer_width)

        # Extract features from the end-of-sequence token (assumed to be at the last position)
        # In practice, you might choose a different strategy to locate the [EOS] token.
        eos_tokens = x[torch.arange(x.shape[0]), -1]  # (batch_size, transformer_width)
        
        # Final normalization and linear projection.
        x = self.ln_final(eos_tokens)
        x = x @ self.text_projection  # (batch_size, embed_dim)
        return x
    

class GeneralTextEncoder(nn.Module):
    def __init__(
        self,
        num_tokens: int = 49152,
        embed_dim: int = 768,
        context_length: int = 512,
        transformer_width: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 12,
        dim_head: int = None,
        use_rotary_emb: bool = False,
        causal: bool = False,
        use_cls_token: bool = False,
        final_output_handling: bool = True,
        dropout: float = 0.1
    ):
        """
        An improved CLIP-style text encoder with optional rotary embeddings,
        an optional classification token, and configurable final output handling.

        Args:
            num_tokens: Vocabulary size.
            embed_dim: Dimensionality of the final output embedding.
            context_length: Maximum sequence length.
            transformer_width: Hidden dimension for the transformer.
            transformer_heads: Number of attention heads.
            transformer_layers: Number of transformer layers.
            dim_head: Dimensionality per attention head (default: transformer_width // transformer_heads).
            use_rotary_emb: Whether to use rotary positional embeddings.
            causal: Whether the transformer operates in causal (autoregressive) mode.
            use_cls_token: Whether to prepend a learnable classification token (only applicable when causal=False).
            final_output_handling: If True, the encoder applies layer normalization and a final projection.
                                     If False, it returns the raw transformer outputs.
            dropout: Dropout rate used in the transformer layers.
        """
        super().__init__()
        if dim_head is None:
            dim_head = transformer_width // transformer_heads

        self.context_length = context_length
        self.token_emb = nn.Embedding(num_tokens, transformer_width)

        # Choose between absolute and rotary positional embeddings.
        if not use_rotary_emb:
            self.positional_emb = nn.Embedding(context_length, transformer_width)
            self.rotary_emb = None
        else:
            self.positional_emb = None
            self.rotary_emb = RotaryEmbedding(min(dim_head, 32))

        # Optionally include a classification token (only if not in causal mode).
        self.use_cls_token = use_cls_token and (not causal)
        if self.use_cls_token:
            # The cls_token is learned and will be prepended to the token embeddings.
            self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_width))
        else:
            self.cls_token = None

        # Build the transformer encoder using PyTorch's built-in modules.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_width,
            nhead=transformer_heads,
            dim_feedforward=transformer_width * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Final output handling: layer norm and projection (optional)
        self.final_output_handling = final_output_handling
        if final_output_handling:
            self.ln_final = nn.LayerNorm(transformer_width)
            self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
            nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)
        else:
            self.ln_final = None
            self.text_projection = None

        self.causal = causal

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the text encoder.

        Args:
            tokens: Tensor of shape (batch, seq_len) containing token indices.
            mask: Optional boolean mask tensor for padding positions (True for positions to keep).

        Returns:
            If final_output_handling is True: A tensor of shape (batch, embed_dim) containing the final text embeddings.
            Otherwise: The raw transformer outputs of shape (batch, seq_len, transformer_width).
        """
        b, n = tokens.shape
        device = tokens.device

        # Embed tokens.
        x = self.token_emb(tokens)  # (b, n, transformer_width)

        # Add positional embeddings.
        if self.positional_emb is not None:
            pos_emb = self.positional_emb(torch.arange(n, device=device))  # (n, transformer_width)
            x = x + pos_emb.unsqueeze(0)

        # Note: If using rotary embeddings, you would typically integrate them into your attention mechanism.
        # Here, we demonstrate obtaining the rotary embeddings, but further integration is required. Default transformer does not support rotary embeddings.
        rotary_pos = None
        if self.rotary_emb is not None:
            rotary_pos = self.rotary_emb(n + (1 if self.cls_token is not None else 0), device=device)
            

        # Optionally prepend a classification token.
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(b, -1, -1)  # (b, 1, transformer_width)
            x = torch.cat((cls_tokens, x), dim=1)
            if mask is not None:
                mask = F.pad(mask, (1, 0), value=True)

        # Prepare the input for the transformer (shape: seq_len, batch, transformer_width).
        x = x.transpose(0, 1)

        # If causal mode is enabled and no mask is provided, create a causal mask.
        if self.causal and mask is None:
            seq_len = x.size(0)
            # Create an upper-triangular mask with True indicating positions to attend.
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        # Pass through the transformer.
        x = self.transformer(x, src_key_padding_mask=mask, )
        x = x.transpose(0, 1)  # (b, seq_len, transformer_width)

        if self.final_output_handling:
            # Determine which token to use for the final representation.
            if self.cls_token is not None:
                # If a cls token was added, use its corresponding output.
                representation = x[:, 0]
            else:
                # Otherwise, assume the last token (e.g. [EOS]) represents the sequence.
                representation = x[:, -1]

            # Apply layer normalization and final projection.
            representation = self.ln_final(representation)
            representation = representation @ self.text_projection  # (b, embed_dim)
            return representation
        else:
            # Return the raw transformer outputs.
            return x
        

class CXRBert():
    def __init__(self):
        self.url = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.tokenizer = AutoTokenizer.from_pretrained(self.url, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.url, trust_remote_code=True)

    def forward(self, text_prompts):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_prompts,
                                                            add_special_tokens=True,
                                                            padding='longest',
                                                            return_tensors='pt')
        embeddings = self.model.get_projected_text_embeddings(input_ids=tokenizer_output.input_ids,
                                                              attention_mask=tokenizer_output.attention_mask)
        return embeddings
    


# Example usage:
if __name__ == "__main__":
    batch_size = 4
    context_length = 76
    dummy_tokens = torch.randint(0, 49152, (batch_size, context_length))
    
    # Create an instance with rotary embeddings, a classification token, and final output handling.
    model = GeneralTextEncoder(
        num_tokens=49152,
        embed_dim=512,
        context_length=context_length,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        use_rotary_emb=True,
        causal=False,
        use_cls_token=True,
        final_output_handling=True
    )
    
    # Forward pass.
    final_embedding = model(dummy_tokens)
    print("Final text embedding shape:", final_embedding.shape)
    
    # Alternatively, to get the raw transformer outputs, disable final output handling.
    model_no_final = GeneralTextEncoder(final_output_handling=False)
    raw_output = model_no_final(dummy_tokens)
    print("Raw transformer output shape:", raw_output.shape)