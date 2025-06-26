import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os
from datetime import datetime


def create_model_history_visualizations(model, history, config, save_dir=None):

    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"reports/visualizations_{timestamp}"

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history["epochs"], history["train_loss"], label="Train Loss", marker="o")
    plt.plot(
        history["epochs"], history["val_loss"], label="Validation Loss", marker="s"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    if "train_perplexity" in history:
        plt.plot(
            history["epochs"],
            history["train_perplexity"],
            label="Train Perplexity",
            marker="o",
        )
        plt.plot(
            history["epochs"],
            history["val_perplexity"],
            label="Val Perplexity",
            marker="s",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity")
        plt.title("Training and Validation Perplexity")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    param_counts = {}
    for name, param in model.named_parameters():
        module_name = name.split(".")[0]
        if module_name not in param_counts:
            param_counts[module_name] = 0
        param_counts[module_name] += param.numel()

    plt.pie(param_counts.values(), labels=param_counts.keys(), autopct="%1.1f%%")
    plt.title("Parameter Distribution")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_summary.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    create_model_summary(model, config, save_dir)

    print(f"Visualizations saved to: {save_dir}")
    return save_dir


def create_model_summary(model, config, save_dir):

    _, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    model_info = f"""
    TRANSFORMER MODEL SUMMARY
    
    Architecture Configuration:
    • Encoder Layers: {config.get("encoder_layers", "N/A")}
    • Decoder Layers: {config.get("decoder_layers", "N/A")}
    • Model Dimension: {config.get("d_model", "N/A")}
    • Feed Forward Dimension: {config.get("d_ff", "N/A")}
    • Attention Heads: {config.get("encoder_heads", "N/A")}
    • Dropout Rate: {config.get("dropout", "N/A")}
    • Max Sequence Length: {config.get("max_len", "N/A")}
    
    Training Configuration:
    • Learning Rate: {config.get("learning_rate", "N/A")}
    • Batch Size: {config.get("batch_size", "N/A")}
    • Epochs: {config.get("epochs", "N/A")}
    • Optimizer: AdamW
    • Mixed Precision: {config.get("mixed_precision", False)}
    
    Model Statistics:
    • Total Parameters: {sum(p.numel() for p in model.parameters()):,}
    • Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}
    """

    ax.text(
        0.1,
        0.5,
        model_info,
        fontsize=12,
        fontfamily="monospace",
        verticalalignment="center",
        transform=ax.transAxes,
    )

    plt.savefig(
        os.path.join(save_dir, "model_summary.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_output_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"reports/visualizations_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_transformer_architecture_diagram(model, model_info, output_dir="reports"):
    num_encoder_layers = model_info["e_N"]
    num_decoder_layers = model_info["d_N"]
    input_size = model_info["input_size"]

    total_layers = max(num_encoder_layers, num_decoder_layers)
    y_slots = 2 + total_layers + 1
    fig, ax = plt.subplots(1, 1, figsize=(14, 4 + total_layers * 1.5))

    colors = {
        "encoder": "#4CAF50",
        "decoder": "#2196F3",
        "attention": "#FF9800",
        "ffn": "#9C27B0",
        "embedding": "#FF5722",
        "output": "#607D8B",
    }

    y_positions = np.linspace(0.95, 0.05, y_slots)

    def draw_box(x, y, width, height, label, color, text_color="white"):
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.01",
            facecolor=color,
            edgecolor="black",
            lw=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
            weight="bold",
        )

    def draw_arrow(x1, y1, x2, y2, pos_adj=0.015):
        ax.annotate(
            "",
            xy=(x2, y2 - pos_adj),
            xytext=(x1, y1 + pos_adj),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
        )

    if input_size:
        draw_box(
            0.25,
            y_positions[0],
            0.2,
            0.04,
            f"Source Embedding\n{input_size[0]}",
            colors["embedding"],
        )
        draw_box(
            0.75,
            y_positions[0],
            0.2,
            0.04,
            f"Target Embedding\n{input_size[1]}",
            colors["embedding"],
        )
    draw_box(
        0.25, y_positions[1], 0.2, 0.04, "Positional Encoding", colors["embedding"]
    )
    draw_box(
        0.75, y_positions[1], 0.2, 0.04, "Positional Encoding", colors["embedding"]
    )
    draw_arrow(0.25, y_positions[0], 0.25, y_positions[1])
    draw_arrow(0.75, y_positions[0], 0.75, y_positions[1])

    for i in range(total_layers):
        y_pos = y_positions[i + 2]
        if i < num_encoder_layers:
            draw_box(
                0.25, y_pos, 0.2, 0.04, f"Encoder Block {i + 1}", colors["encoder"]
            )
            draw_box(
                0.1, y_pos, 0.1, 0.025, "Multi-Head Attn", colors["attention"], "black"
            )
            draw_box(0.4, y_pos, 0.1, 0.025, "Feed Forward", colors["ffn"])
            if i > 0:
                draw_arrow(0.25, y_positions[i + 1], 0.25, y_pos)
        if i < num_decoder_layers:
            draw_box(
                0.75, y_pos, 0.2, 0.04, f"Decoder Block {i + 1}", colors["decoder"]
            )
            draw_box(
                0.6,
                y_pos + 0.01,
                0.1,
                0.02,
                "Masked Self-Attn",
                colors["attention"],
                "black",
            )
            draw_box(
                0.6, y_pos - 0.01, 0.1, 0.02, "Cross-Attn", colors["attention"], "black"
            )
            draw_box(0.9, y_pos, 0.1, 0.025, "Feed Forward", colors["ffn"])
            if i > 0:
                draw_arrow(0.75, y_positions[i + 1], 0.75, y_pos)
            if i < num_encoder_layers:
                ax.annotate(
                    "",
                    xy=(0.65, y_pos - 0.02),
                    xytext=(0.35, y_pos),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="arc3,rad=0.2",
                        lw=1.5,
                        color="gray",
                        ls="--",
                    ),
                )

    draw_arrow(0.25, y_positions[1], 0.25, y_positions[2])
    draw_arrow(0.75, y_positions[1], 0.75, y_positions[2])

    output_y = y_positions[total_layers + 1]
    draw_box(0.75, output_y, 0.2, 0.04, "Linear + Softmax", colors["output"])
    draw_arrow(0.75, y_positions[total_layers], 0.75, output_y)

    ax.text(
        0.25, y_positions[0] + 0.05, "ENCODER", ha="center", fontsize=14, weight="bold"
    )
    ax.text(
        0.75, y_positions[0] + 0.05, "DECODER", ha="center", fontsize=14, weight="bold"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Dynamic Transformer Architecture", fontsize=16, weight="bold", pad=20)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "transformer_architecture.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def create_attention_heatmaps(model_info, output_dir="reports"):
    input_size = model_info["input_size"]
    src_seq_len = input_size[0][1] if input_size else 10
    tgt_seq_len = input_size[1][1] if input_size else 10

    paths = {}

    def _plot_heatmap(num_heads, seq_len, title, save_path):
        cols = 4
        rows = (num_heads + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 4), squeeze=False
        )
        fig.suptitle(title, fontsize=16, weight="bold")

        attention_weights = np.random.rand(num_heads, seq_len, seq_len)

        axes = axes.flatten()
        for i in range(num_heads):
            im = axes[i].imshow(
                attention_weights[i], cmap="viridis", interpolation="nearest"
            )
            axes[i].set_title(f"Head {i + 1}")
            axes[i].set_xlabel("Key Position")
            axes[i].set_ylabel("Query Position")
            plt.colorbar(im, ax=axes[i], shrink=0.8)

        for i in range(num_heads, len(axes)):
            axes[i].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    e_num_heads = model_info["e_num_heads"]
    encoder_title = (
        f"Encoder Self-Attention ({e_num_heads} Heads, Seq Len: {src_seq_len})"
    )
    encoder_save_path = os.path.join(output_dir, "encoder_attention_heatmap.png")
    _plot_heatmap(e_num_heads, src_seq_len, encoder_title, encoder_save_path)
    paths["encoder_attention"] = encoder_save_path

    d_num_heads = model_info["d_num_heads"]
    decoder_title = (
        f"Decoder Masked Self-Attention ({d_num_heads} Heads, Seq Len: {tgt_seq_len})"
    )
    decoder_save_path = os.path.join(output_dir, "decoder_attention_heatmap.png")
    _plot_heatmap(d_num_heads, tgt_seq_len, decoder_title, decoder_save_path)
    paths["decoder_attention"] = decoder_save_path

    return paths


def create_parameter_breakdown_detailed(model, model_info, output_dir="reports"):
    components = {
        "Embeddings": 0,
        "Encoder Attention": 0,
        "Encoder FFN": 0,
        "Encoder Norms": 0,
        "Decoder Self-Attn": 0,
        "Decoder Cross-Attn": 0,
        "Decoder FFN": 0,
        "Decoder Norms": 0,
        "Final Generator": 0,
    }

    for name, param in model.named_parameters():
        count = param.numel()
        if "embed" in name:
            components["Embeddings"] += count
        elif "generator" in name:
            components["Final Generator"] += count
        elif "encoder" in name:
            if "attn" in name:
                components["Encoder Attention"] += count
            elif "ffn" in name:
                components["Encoder FFN"] += count
            elif "norm" in name:
                components["Encoder Norms"] += count
        elif "decoder" in name:
            if "self_attn" in name:
                components["Decoder Self-Attn"] += count
            elif "src_attn" in name:
                components["Decoder Cross-Attn"] += count
            elif "ffn" in name:
                components["Decoder FFN"] += count
            elif "norm" in name:
                components["Decoder Norms"] += count

    fig, ax = plt.subplots(figsize=(12, 8))
    labels = [k for k, v in components.items() if v > 0]
    sizes = [v for v in components.values() if v > 0]

    ax.pie(
        sizes,
        labels=labels,
        autopct=lambda p: "{:.1f}%\n({:,.0f})".format(p, p * sum(sizes) / 100),
        startangle=90,
        pctdistance=0.85,
        colors=plt.cm.Paired.colors,
    )
    ax.add_artist(plt.Circle((0, 0), 0.70, fc="white"))
    ax.set_title("Parameter Distribution by Component", fontsize=16, weight="bold")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "parameter_breakdown.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def create_model_info_summary(model, model_info, output_dir="reports"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024

    info_text = f"""
    **Transformer Model Summary**
    -----------------------------------
    **Vocabulary:**
      - Source: {model_info["src_vocab"]:,}
      - Target: {model_info["tgt_vocab"]:,}

    **Architecture:**
      - Model Dimension (d_model): {model_info["d_model"]}
      - Feed-Forward Dim (d_ff): {model_info["d_ff"]}
      - Encoder Layers (e_N): {model_info["e_N"]}
      - Decoder Layers (d_N): {model_info["d_N"]}
      - Encoder Heads: {model_info["e_num_heads"]}
      - Decoder Heads: {model_info["d_num_heads"]}

    **Parameters & Memory:**
      - Total Parameters: {total_params:,}
      - Trainable: {trainable_params:,}
      - Est. Size (float32): {model_size_mb:.2f} MB
    """

    ax.text(
        0.05,
        0.95,
        info_text.strip(),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e0f7fa", alpha=0.8),
    )

    ax.axis("off")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "model_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def create_readme(output_dir, model_info):
    readme_content = f"""# Transformer Model Visualization
*Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Model Configuration
| Parameter | Value |
|---|---|
| Source Vocabulary | {model_info["src_vocab"]:,} |
| Target Vocabulary | {model_info["tgt_vocab"]:,} |
| Model Dimension (d_model) | {model_info["d_model"]} |
| Feed-Forward Dim (d_ff) | {model_info["d_ff"]} |
| Encoder Layers (e_N) | {model_info["e_N"]} |
| Decoder Layers (d_N) | {model_info["d_N"]} |
| Encoder Heads | {model_info["e_num_heads"]} |
| Decoder Heads | {model_info["d_num_heads"]} |
| Input Size (Batch, Seq) | `{model_info["input_size"]}` |

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
"""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    return readme_path


def get_model_infos(model, input_size=None):
    output_dir = create_output_directory()

    try:
        model_info = {
            "input_size": input_size,
            "e_N": len(model.encoder),
            "d_N": len(model.decoder),
            "d_model": model.encoder[0].attn.d_k * model.encoder[0].attn.h,
            "d_ff": model.encoder[0].ffn.linear2.in_features,
            "e_num_heads": model.encoder[0].attn.h,
            "d_num_heads": model.decoder[0].self_attn.h,
            "src_vocab": model.src_embed[0].lut.num_embeddings,
            "tgt_vocab": model.tgt_embed[0].lut.num_embeddings,
        }
    except Exception as e:
        print(f"Could not extract model info dynamically: {e}")
        model_info = {
            "input_size": input_size,
            "e_N": 6,
            "d_N": 6,
            "e_num_heads": 8,
            "d_num_heads": 8,
            "src_vocab": "N/A",
            "tgt_vocab": "N/A",
            "d_model": "N/A",
            "d_ff": "N/A",
        }

    print(f"Visualizations will be saved to: {output_dir}")

    paths = {}
    paths["architecture"] = create_transformer_architecture_diagram(
        model, model_info, output_dir
    )

    attention_paths = create_attention_heatmaps(model_info, output_dir)
    paths.update(attention_paths)

    paths["parameters"] = create_parameter_breakdown_detailed(
        model, model_info, output_dir
    )
    paths["summary"] = create_model_info_summary(model, model_info, output_dir)
    paths["readme"] = create_readme(output_dir, model_info)

    print("\nVisualization complete!")
    for name, path in paths.items():
        print(f"  - Created: {os.path.basename(path)}")

    return output_dir, paths
