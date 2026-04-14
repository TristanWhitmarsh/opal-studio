import numpy as np
import traceback
from instanseg import InstanSeg
try:
    model = InstanSeg("fluorescence_nuclei")
    x = np.random.rand(100, 100).astype(np.float32)
    # try 1 channel
    out, img = model.eval_small_image(x, 0.5)
    print("1-channel (fluorescence_nuclei) OUT TYPE:", type(out))
    print("1-channel (fluorescence_nuclei) OUT 0 SHAPE:", out[0].shape)
    
except Exception as e:
    traceback.print_exc()

import torch
# let's try with 3 channels for brightfield
try:
    model2 = InstanSeg("brightfield_nuclei")
    x3 = np.random.rand(100, 100, 3).astype(np.float32)
    out2, img2 = model2.eval_small_image(x3, 0.5)
    print("3-channel (brightfield) OUT 0 SHAPE:", out2[0].shape)
    if isinstance(out2[0], torch.Tensor):
        print("Tensor output")
except Exception as e:
    traceback.print_exc()
