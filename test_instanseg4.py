import numpy as np
import traceback
from instanseg import InstanSeg
try:
    model = InstanSeg("fluorescence_nuclei_and_cells")
    x = np.random.rand(100, 100).astype(np.float32)
    # try 1 channel
    out, img = model.eval_small_image(x, 0.5)
    print("1-channel (fluorescence_nuclei_and_cells) OUT LENGTH:", len(out))
    print("1-channel (fluorescence_nuclei_and_cells) OUT 0 SHAPE:", out[0].shape)
    if len(out) > 1:
        print("1-channel (fluorescence_nuclei_and_cells) OUT 1 SHAPE:", out[1].shape)
except Exception as e:
    traceback.print_exc()
