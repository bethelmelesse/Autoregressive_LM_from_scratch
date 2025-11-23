import torch
import torch.nn as nn


class TransformerDecoderBlock(nn.Module):
    """Transformer decoder block (decoder-only architecture)."""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the transformer decoder block.

        Args:
            d_model (int): Dimension of the model's hidden states. Default: 512
            num_heads (int): Number of attention heads. Default: 8
            d_ff (int): Dimension of the feed-forward network's hidden layer. Default: 2048
            dropout (float): Dropout probability applied throughout the block. Default: 0.1
        """
        super().__init__()

        # using pytorch's library
        # if batch_first is true, input is given in (batch_size, sequence_length, d_model)
        self.multihead_att = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input: torch.Tensor,
        attn_mask: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass of the transformer decoder block.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model)
            attn_mask (torch.Tensor): Attention mask of shape (sequence_length, sequence_length)
                 or (batch_size * num_heads, sequence_length, sequence_length).
            padding_mask (torch.Tensor): Padding mask of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model)
        """
        # Multi-head attention - Shape: (batch_size, sequence_length, d_model)
        attn_output, _ = self.multihead_att(
            query=input,
            key=input,
            value=input,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
        )

        # Dropout + Residual + LayerNorm
        layernorm_output = self.layer_norm1(input + self.dropout(attn_output))

        # Feedforward network
        ff_output = self.ffn(layernorm_output)

        # Dropout + Residual + LayerNorm
        output = self.layer_norm2(layernorm_output + self.dropout(ff_output))
        return output  # Shape: (batch_size, sequence_length, d_model)


def main() -> None:
    """Test the transformer decoder block."""
    print("\nTesting Transformer Deconding Block...")
    # (batch_size, sequence_length, d_model)
    input = torch.randn(1, 10, 512)
    attn_mask = None
    key_padding_mask = None
    transformer_decoder_block = TransformerDecoderBlock()
    output = transformer_decoder_block(input, attn_mask, key_padding_mask)
    print(output.shape)
    print()


if __name__ == "__main__":
    main()
