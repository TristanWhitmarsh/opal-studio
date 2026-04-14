import numpy as np
from instanseg import InstanSeg
try:
    model = InstanSeg("brightfield_nuclei")
    x = np.random.rand(100, 100).astype(np.float32)
    out, img = model.eval_small_image(x, 0.5)
    print("Type of out:", type(out))
    if hasattr(out, 'shape'):
        print("Out shape:", out.shape)
    elif isinstance(out, tuple) or isinstance(out, list):
        print("Out 0 shape:", out[0].shape)
    import torch
    if isinstance(out[0], torch.Tensor):
        print("Out 0 is tensor")
except Exception as e:
    print(e)
