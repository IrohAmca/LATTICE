from models.transformer import Transformer
from utils.visualization.torch_viz import main


model = Transformer(
    src_vocab=1000,
    tgt_vocab=1000,
    e_N=6,
    d_N=6,
    d_model=512,
    d_ff=2048,
    e_num_heads=8,
    d_num_heads=8,
    dropout=0.1,
    max_len=100,
)

main(model=model, input_size=[(1, 10), (1, 10)])
