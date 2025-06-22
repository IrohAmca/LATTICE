# Transformer Model Visualization
*Generated on: 2025-06-23 00:32:06*

## Model Configuration
| Parameter | Value |
|---|---|
| Source Vocabulary | 1,000 |
| Target Vocabulary | 1,000 |
| Model Dimension (d_model) | 512 |
| Feed-Forward Dim (d_ff) | 2048 |
| Encoder Layers (e_N) | 6 |
| Decoder Layers (d_N) | 6 |
| Encoder Heads | 8 |
| Decoder Heads | 8 |
| Input Size (Batch, Seq) | `[(1, 10), (1, 10)]` |

## Visualizations

### Architecture
![Architecture Diagram](transformer_architecture.png)

### Parameter Breakdown
![Parameter Breakdown](parameter_breakdown.png)

### Attention Heads

#### Encoder Self-Attention
![Encoder Attention](encoder_attention_heatmap.png)

#### Decoder Masked Self-Attention
![Decoder Attention](decoder_attention_heatmap.png)

### Model Summary
![Model Summary](model_summary.png)
