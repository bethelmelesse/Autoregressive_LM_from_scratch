"""Input embedder for the autoregressive language model.

This module is used to embed the input tokens and positional encodings into a single
embedding.
"""

import torch
from torch import nn


class Embedder(nn.Module):
    """Input embedding with learned positional encoding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the input embedder.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimension of the model.
            max_seq_len (int): The maximum sequence length.
            dropout (float): The dropout rate.
        """
        super().__init__()

        # Token embedding
        self.token_embedder = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )

        # Learned positional embedding
        self.positional_embedder = nn.Embedding(
            num_embeddings=max_seq_len, embedding_dim=d_model
        )

        self.scale = d_model**0.5  #  Scaling factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass of the input embedder.

        Args:
            input_ids (torch.Tensor): The input token ids. Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: The input embeddings. Shape: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings (batch_size, seq_len, d_model)
        token_embeds = self.token_embedder(input_ids) * self.scale

        # Position IDs: [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)  # (1, seq_len)

        # Positional embeddings (batch_size, seq_len, d_model)
        pos_embeds = self.positional_embedder(position_ids)

        # Combine token embeddings and positional embedding
        # shape: (batch_size, seq_len, d_model)
        embeddings = self.dropout(token_embeds + pos_embeds)

        return embeddings


def main() -> None:
    """Test the input embedder."""
    print("\nTesting Embedder...")
    # Sample input ids
    input_ids = torch.randint(low=0, high=1000, size=(1, 10))
    print(f"Input IDs shape: {input_ids.shape}")

    # Initialize the input embedder
    input_embedder = Embedder(
        vocab_size=1000, d_model=512, max_seq_len=512, dropout=0.1
    )
    embeddings = input_embedder(input_ids)
    print(f"Embeddings shape: {embeddings.shape}\n")


if __name__ == "__main__":
    main()
