import torch as T
from tools.transformers.transformer import TransformerEncoder

def test_attention(model_dim, num_layers, n_heads, unpadded, device="cuda"):
    # check device
    if T.cuda.is_available():
        device = "cuda"

    # init transformer
    transformer = TransformerEncoder(model_dim=model_dim, num_layers=num_layers, mha_config={"n_heads": n_heads}, unpadded=unpadded, device=device)
    
    dtype = T.float16 if device=="cuda" else T.float32

    for _ in range(5):

        with T.autocast(device_type=device, dtype=dtype):
            # init random data
            pc_input = T.rand(512,20, model_dim, device=device, dtype=dtype)

            pc_mask = T.rand(512, 20, device=device) > 0.5

            # run transformer
            out = transformer(pc_input, mask_vk=pc_mask)
            
            # check for NaN
            if out.isnan().any():
                raise ValueError(f"NaN in output of nr {_}")
    
    return out

if __name__ == "__main__":
    test_attention(model_dim=64, num_layers=2, n_heads=4, unpadded=True)
    