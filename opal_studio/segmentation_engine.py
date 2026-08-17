import os
import sys
import traceback
import numpy as np
from pathlib import Path
import importlib.util

def ensure_peakdetect_scipy_compat():
    spec = importlib.util.find_spec("peakdetect")
    if spec is None or not spec.origin:
        return

    peakdetect_py = Path(spec.origin).resolve().parent / "peakdetect.py"
    if not peakdetect_py.exists():
        return

    text = peakdetect_py.read_text(encoding="utf-8")

    replacements = {
        "from scipy import fft, ifft": "from scipy.fftpack import fft, ifft",
        "from scipy.fft import fft, ifft": "from scipy.fftpack import fft, ifft",
    }

    new_text = text
    for old, new in replacements.items():
        new_text = new_text.replace(old, new)

    if new_text != text:
        peakdetect_py.write_text(new_text, encoding="utf-8")

def ensure_instanseg_py39_compat():
    """
    Work around instanseg-torch 0.1.1 on Python 3.9 by ensuring
    `from __future__ import annotations` is present at the top of
    instanseg/utils/tiling.py before InstaSeg is imported.
    """
    spec = importlib.util.find_spec("instanseg")
    if spec is None or not spec.origin:
        return

    instanseg_dir = Path(spec.origin).resolve().parent
    tiling_py = instanseg_dir / "utils" / "tiling.py"
    if not tiling_py.exists():
        return

    wanted = "from __future__ import annotations\n"
    text = tiling_py.read_text(encoding="utf-8")

    if not text.startswith(wanted):
        tiling_py.write_text(wanted + text, encoding="utf-8")


def _postprocess_labels(labels, fill_holes=False, keep_largest=False):
    """
    Optional post-processing for label images.
    fill_holes: fills internal holes in each object mask.
    keep_largest: if an object ID is fragmented, only the largest component is kept.
    """
    if not (fill_holes or keep_largest):
        return labels
    
    from scipy.ndimage import binary_fill_holes, label as cc_label
    new_labels = np.zeros_like(labels)
    unique_ids = np.unique(labels)
    
    for idx in unique_ids:
        if idx == 0: continue
        mask = (labels == idx)
        
        if fill_holes:
            mask = binary_fill_holes(mask)
            
        if keep_largest:
            labeled_comp, num = cc_label(mask)
            if num > 1:
                sizes = np.bincount(labeled_comp.ravel())
                largest = sizes[1:].argmax() + 1
                mask = (labeled_comp == largest)
        
        new_labels[mask] = idx
    return new_labels

def run_segmentation_task_pipe(conn, params, input_channels_data, stop_event=None):
    """
    Worker function to run segmentation in a separate process.
    This prevents DLL conflicts between TensorFlow and PyTorch.
    """
    try:
        method = params.get("method")
        results = []

        if method == "stardist":
            from stardist.models import StarDist2D
            model_folder = params.get("model_folder", params["model_name"])
            # Ensure we look in the correct directory relative to the app
            basedir = os.path.join(os.path.dirname(__file__), "models", "stardist")
            
            if os.path.isdir(os.path.join(basedir, model_folder)):
                model = StarDist2D(None, name=model_folder, basedir=basedir)
            else:
                model = StarDist2D.from_pretrained(params["model_name"])

            kwargs = {"nms_thresh": params["nms_thresh"]}
            if not params["use_default_thresh"]:
                kwargs["prob_thresh"] = params["prob_thresh"]
            
            x = input_channels_data[0]
            # Tiling for large images
            n_tiles = None
            if x.shape[0] > 1024 or x.shape[1] > 1024:
                n_tiles = (int(np.ceil(x.shape[0] / 1024)), int(np.ceil(x.shape[1] / 1024)))
            
            labels, _ = model.predict_instances(x, n_tiles=n_tiles, **kwargs)
            results.append((labels, params.get("override_name", "StarDist"), False))

        elif method == "cellpose":
            from cellpose import models
            import torch
            use_gpu = torch.cuda.is_available()
            
            if params.get("model_path"):
                model = models.CellposeModel(pretrained_model=params["model_path"], gpu=use_gpu)
            else:
                model = models.Cellpose(model_type=params["model_name"], gpu=use_gpu)
            
            x = input_channels_data[0]
            res = model.eval([x], 
                             diameter=params.get("diameter"), 
                             channels=[[0, 0]], 
                             batch_size=64,
                             cellprob_threshold=params.get("cellprob_threshold", 0.0),
                             flow_threshold=params.get("flow_threshold", 0.4))
            labels = res[0][0]
            results.append((labels, params.get("override_name", "Cellpose"), False))

        elif method == "omnipose":
            ensure_peakdetect_scipy_compat()
            from cellpose_omni import models
            import torch
            use_gpu = torch.cuda.is_available()
            
            if params.get("model_path"):
                model = models.CellposeModel(pretrained_model=params["model_path"], gpu=use_gpu, nchan=2, nclasses=4, omni=True)
            else:
                model = models.CellposeModel(model_type=params["model_name"], gpu=use_gpu, nchan=2, nclasses=4, omni=True)
            
            x = input_channels_data[0]
            # cellpose_omni eval arguments are similar to cellpose
            res = model.eval([x], 
                             diameter=params.get("diameter"), 
                             channels=[[0, 0]], 
                             batch_size=64, 
                             omni=True,
                             mask_threshold=params.get("mask_threshold", 0.0),
                             flow_threshold=params.get("flow_threshold", 0.4))
            labels = res[0][0]
            results.append((labels, params.get("override_name", "Omnipose"), False))

        elif method == "instanseg":
            import torch
            ensure_instanseg_py39_compat()
            
            model_name = params["model_name"]
            model_path = params.get("model_path")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            x = input_channels_data[0]
            
            if model_path and os.path.exists(model_path):
                from instanseg.utils.loss.instanseg_loss import InstanSeg as InstanSegPostProcessor
                from instanseg.utils.model_loader import load_model
                from instanseg.utils.utils import percentile_normalize
                import torch.nn.functional as F

                result = load_model(str(model_path))
                backbone = result[0] if isinstance(result, (tuple, list)) else result
                backbone.eval()
                backbone = backbone.to(device)

                postprocessor = InstanSegPostProcessor(
                    n_sigma=2, dim_coords=2, dim_seeds=1,
                    cells_and_nuclei=False, device=device,
                )
                postprocessor.initialize_pixel_classifier(backbone)
                postprocessor.pixel_classifier = postprocessor.pixel_classifier.to(device)

                n_in = 1
                for name, module in backbone.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        n_in = module.in_channels
                        break

                raw = x
                if raw.ndim == 2 and n_in > 1:
                    raw_input = np.stack([raw] * n_in, axis=0)
                elif raw.ndim == 2:
                    raw_input = raw[np.newaxis, ...]
                else:
                    raw_input = raw

                TILE_SIZE = 128
                OVERLAP = 16
                H, W = raw_input.shape[-2], raw_input.shape[-1]

                def _run_tile(patch_np):
                    t = torch.from_numpy(patch_np).unsqueeze(0).float()
                    t = torch.stack([percentile_normalize(t[0])]).to(device)
                    ph, pw = t.shape[-2], t.shape[-1]
                    pad_h = (32 - ph % 32) % 32
                    pad_w = (32 - pw % 32) % 32
                    if pad_h or pad_w:
                        pad_mode = 'reflect' if ph > pad_h and pw > pad_w else 'constant'
                        t = F.pad(t, (0, pad_w, 0, pad_h), mode=pad_mode)
                    with torch.no_grad():
                        emb = backbone(t)
                    if pad_h or pad_w:
                        emb = emb[..., :ph, :pw]
                    with torch.no_grad():
                        inst = postprocessor.postprocessing(
                            emb[0], device=device, classifier=postprocessor.pixel_classifier,
                        )
                    return inst[0].cpu().numpy().astype(np.int32)

                if H <= TILE_SIZE and W <= TILE_SIZE:
                    cell_mask = _run_tile(raw_input)
                else:
                    cell_mask = np.zeros((H, W), dtype=np.int32)
                    next_id = 1
                    stride = TILE_SIZE - OVERLAP
                    ys = list(range(0, H, stride))
                    xs = list(range(0, W, stride))
                    n_tiles = len(ys) * len(xs)
                    tile_counter = 0
                    offset_y = params.get("crop_offset_y", 0)
                    offset_x = params.get("crop_offset_x", 0)
                    full_shape = params.get("full_shape") or [H, W]
                    for y0 in ys:
                        for x0 in xs:
                            if stop_event is not None and stop_event.is_set():
                                conn.send({"type": "cancelled"})
                                return
                            y1 = min(y0 + TILE_SIZE, H)
                            x1 = min(x0 + TILE_SIZE, W)
                            patch = raw_input[..., y0:y1, x0:x1]
                            tile_labels = _run_tile(patch)
                            # Only keep cells whose centroid falls in the non-overlap
                            # inner region to avoid duplicates at tile borders.
                            inner_y0 = OVERLAP // 2 if y0 > 0 else 0
                            inner_x0 = OVERLAP // 2 if x0 > 0 else 0
                            inner_y1 = (y1 - y0) - (OVERLAP // 2) if y1 < H else (y1 - y0)
                            inner_x1 = (x1 - x0) - (OVERLAP // 2) if x1 < W else (x1 - x0)
                            for obj_id in np.unique(tile_labels):
                                if obj_id == 0:
                                    continue
                                mask = tile_labels == obj_id
                                ys_obj, xs_obj = np.where(mask)
                                cy = int(np.mean(ys_obj))
                                cx = int(np.mean(xs_obj))
                                if inner_y0 <= cy < inner_y1 and inner_x0 <= cx < inner_x1:
                                    cell_mask[y0 + ys_obj, x0 + xs_obj] = next_id
                                    next_id += 1
                            conn.send({
                                "type": "tile_update",
                                "tile_idx": tile_counter,
                                "n_tiles": n_tiles,
                                "y0": y0 + offset_y,
                                "x0": x0 + offset_x,
                                "y1": y1 + offset_y,
                                "x1": x1 + offset_x,
                                "tile_labels": cell_mask[y0:y1, x0:x1].copy(),
                                "full_shape": full_shape,
                                "name": params.get("override_name", "InstanSeg"),
                                "is_cell_mask": False,
                                "target_mode": params.get("target_mode", "new"),
                                "target_mask_index": params.get("target_mask_index"),
                            })
                            tile_counter += 1

                    print(f"[InstanSeg] All {n_tiles} tiles processed. "
                          f"Stitching & post-processing…", flush=True)

                cell_mask = _postprocess_labels(
                    cell_mask,
                    fill_holes=params.get("fill_holes", False),
                    keep_largest=params.get("keep_largest", False)
                )
                print(f"[InstanSeg] Post-processing complete "
                      f"({int(cell_mask.max())} objects). Sending result…", flush=True)

                results.append((cell_mask, f"InstanSeg ({model_name})", False))
                
            else:
                from instanseg import InstanSeg
                local_model_dir = os.path.join(os.path.dirname(__file__), "models", "instanseg", model_name)
                if os.path.exists(os.path.join(local_model_dir, "instanseg.pt")):
                    model_name = local_model_dir
                
                model = InstanSeg(model_name, device=device)
                
                x_input = x
                if "brightfield" in params["model_name"].lower() and x.ndim == 2:
                    x_input = np.stack([x]*3, axis=-1)
                
                if max(x.shape[0], x.shape[1]) > 512:
                    print(f"[InstanSeg] Running '{model_name}' on a "
                          f"{x.shape[0]}×{x.shape[1]} image (tiled, no per-tile "
                          f"progress)… this may take a while.", flush=True)
                    out, _ = model.eval_medium_image(x_input, params.get("pixel_size", 1.0), tile_size=512, batch_size=16)
                else:
                    print(f"[InstanSeg] Running '{model_name}' on a "
                          f"{x.shape[0]}×{x.shape[1]} image…", flush=True)
                    out, _ = model.eval_small_image(x_input, params.get("pixel_size", 1.0))
                print("[InstanSeg] Inference complete. Building result…", flush=True)
                
                labels_tensor = out[0]
                if hasattr(labels_tensor, 'cpu'):
                    labels_tensor = labels_tensor.cpu().numpy()
                elif hasattr(labels_tensor, 'numpy'):
                    labels_tensor = labels_tensor.numpy()
                
                labels_tensor = np.squeeze(labels_tensor)
                
                # Apply post-processing to each channel if needed
                if params.get("fill_holes") or params.get("keep_largest"):
                    if labels_tensor.ndim == 3:
                        for i in range(labels_tensor.shape[0]):
                            labels_tensor[i] = _postprocess_labels(
                                labels_tensor[i], 
                                fill_holes=params.get("fill_holes"),
                                keep_largest=params.get("keep_largest")
                            )
                    else:
                        labels_tensor = _postprocess_labels(
                            labels_tensor, 
                            fill_holes=params.get("fill_holes"),
                            keep_largest=params.get("keep_largest")
                        )

                if labels_tensor.ndim == 3:
                    # First channel is nuclei, rest are cells
                    results.append((labels_tensor[0], "InstanSeg Nuclei", False))
                    for c in range(1, labels_tensor.shape[0]):
                        results.append((labels_tensor[c], "InstanSeg Cells", True))
                else:
                    results.append((labels_tensor, "InstanSeg", False))

        elif method == "mesmer":
            import tensorflow as tf
            from deepcell.applications import Mesmer
            
            if params.get("api_key"):
                os.environ["DEEPCELL_ACCESS_TOKEN"] = params["api_key"]
            
            n_data = input_channels_data[0]
            m_data = np.zeros_like(n_data)  # default: zero-filled membrane
            if len(input_channels_data) > 1:
                m_data = input_channels_data[1]

            input_stack = np.stack([n_data, m_data], axis=-1)
            input_stack = np.expand_dims(input_stack, axis=0)  # (1, H, W, 2)

            model_path = params.get("model_path")
            if model_path and os.path.exists(model_path):
                # Local single-channel nuclear PanopticNet model.
                # It has 2 output heads (inner-distance, outer-distance) and expects
                # (None, 256, 256, 1) input — incompatible with Mesmer's 4-output / 2-channel
                # pipeline.  We run the full inference pipeline manually instead:
                #   histogram_normalization → 256×256 overlapping tiles → stitch → deep_watershed
                from deepcell import layers as dc_layers
                from deepcell_toolbox import histogram_normalization
                from deepcell_toolbox.deep_watershed import deep_watershed

                custom_objects = {
                    name: getattr(dc_layers, name)
                    for name in dir(dc_layers)
                    if not name.startswith("_") and isinstance(getattr(dc_layers, name), type)
                }
                keras_model = tf.keras.models.load_model(
                    model_path, compile=False, custom_objects=custom_objects
                )

                TILE = 256
                OVERLAP = 64  # half-tile overlap to avoid boundary artefacts
                STRIDE = TILE - OVERLAP

                # Preprocess: histogram_normalization on (1, H, W, 1) matching training
                nuc_4d = np.expand_dims(n_data, axis=(0, -1)).astype(np.float32)
                nuc_norm = histogram_normalization(nuc_4d)  # (1, H, W, 1)

                H, W = n_data.shape[:2]

                # Pad so every tile is exactly TILE×TILE
                pad_h = (TILE - H % STRIDE) % STRIDE
                pad_w = (TILE - W % STRIDE) % STRIDE
                nuc_pad = np.pad(nuc_norm, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
                PH, PW = nuc_pad.shape[1], nuc_pad.shape[2]

                # Collect tiles
                tiles, positions = [], []
                for y in range(0, PH - TILE + 1, STRIDE):
                    for x in range(0, PW - TILE + 1, STRIDE):
                        tiles.append(nuc_pad[0, y:y+TILE, x:x+TILE, :])
                        positions.append((y, x))

                tiles_arr = np.stack(tiles, axis=0)  # (N, 256, 256, 1)
                batch_size = params.get("batch_size", 16)
                preds = keras_model.predict(tiles_arr, batch_size=batch_size, verbose=0)
                # preds is a list of 2 arrays each (N, 256, 256, 1): [inner_dist, outer_dist]
                if isinstance(preds, (list, tuple)):
                    n_heads = len(preds)
                else:
                    # single-output model (unlikely but handle gracefully)
                    preds = [preds]
                    n_heads = 1

                # Stitch each head back to full image size with averaging in overlap regions
                stitched_heads = []
                for h_idx in range(n_heads):
                    acc = np.zeros((PH, PW, preds[h_idx].shape[-1]), dtype=np.float32)
                    cnt = np.zeros((PH, PW, 1), dtype=np.float32)
                    for i, (y, x) in enumerate(positions):
                        acc[y:y+TILE, x:x+TILE] += preds[h_idx][i]
                        cnt[y:y+TILE, x:x+TILE] += 1.0
                    cnt = np.maximum(cnt, 1.0)
                    stitched = (acc / cnt)[:H, :W]        # (H, W, C)
                    stitched_heads.append(np.expand_dims(stitched, 0))  # (1, H, W, C)

                # deep_watershed: maxima_index=0 (inner-distance), interior_index=-1 (last head)
                ws_kw = params.get("watershed_kwargs", {})
                label_map = deep_watershed(
                    stitched_heads,
                    radius=ws_kw.get("radius", 3),
                    maxima_threshold=ws_kw.get("maxima_threshold", 0.1),
                    maxima_smooth=ws_kw.get("maxima_smooth", 0),
                    interior_threshold=ws_kw.get("interior_threshold", 0.1),
                    interior_smooth=ws_kw.get("interior_smooth", 2),
                    small_objects_threshold=ws_kw.get("small_objects_threshold", 15),
                    fill_holes_threshold=ws_kw.get("fill_holes_threshold", 15),
                    exclude_border=ws_kw.get("exclude_border", False),
                )
                # deep_watershed returns (1, H, W, 1) — single nuclear mask
                labels = np.squeeze(label_map[0, ..., 0]).astype(np.int32)
                results.append((labels, "Mesmer Nuclei", False))
            else:
                ws_kw = params.get("watershed_kwargs", {})
                app = Mesmer()
                labeled_combined = app.predict(
                    input_stack,
                    image_mpp=params.get("pixel_size", 1.0),
                    batch_size=16,
                    compartment=params.get("compartment", "nuclear"),
                    postprocess_kwargs_nuclear=ws_kw,
                    postprocess_kwargs_whole_cell=ws_kw,
                )
                if labeled_combined.shape[-1] >= 2:
                    cell_labels = np.squeeze(labeled_combined[0, ..., 0]).astype(np.int32)
                    nuc_labels = np.squeeze(labeled_combined[0, ..., 1]).astype(np.int32)
                    results.append((nuc_labels, "Mesmer Nuclei", False))
                    results.append((cell_labels, "Mesmer Cells", True))
                else:
                    labels = np.squeeze(labeled_combined[0, ..., 0]).astype(np.int32)
                    results.append((labels, "Mesmer", False))

        conn.send({"type": "result", "results": results})
    except Exception as e:
        conn.send({"type": "error", "error": str(e), "traceback": traceback.format_exc()})
    finally:
        conn.close()

def run_positivity_task(queue, params, data):
    """
    Worker function for AI cell positivity detection.

    Mirrors Inference.ipynb of CellMarkerPositivity: for every cell and every
    marker channel the model is handed a 64x64 crop centred on the cell holding
    the target channel with N_SIDE neighbouring metal channels on each side,
    the binary mask of the target cell and the binary mask of the other cells.
    """
    try:
        import tensorflow as tf
        from scipy import ndimage

        labels = data["labels"]
        markers = data["markers"]

        # Take the GPU as it is needed rather than reserving all of it up front,
        # so a card shared with the rest of the application still works.
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        model_path = os.path.join(os.path.dirname(__file__), "models", "cellpos", "marker_cnn_epoch_200.h5")
        model = tf.keras.models.load_model(model_path, compile=False)

        # The checkpoint records how it was trained. The marker channels are an
        # odd number, so the channel count alone says whether the mask of the
        # other cells is there as well.
        crop = int(model.input_shape[1])
        n_channels = int(model.input_shape[-1])
        use_other_cells = n_channels % 2 == 1
        n_markers = n_channels - 1 - int(use_other_cells)
        n_side = (n_markers - 1) // 2
        half = crop // 2

        threshold = float(params.get("threshold", 0.5))
        batch_size = int(params.get("batch_size", 512))

        S = markers.shape[2]

        # Opal Studio hands over a label map, which is what the notebook's
        # connected component analysis reconstructs from its binary mask. Using
        # it directly keeps touching cells apart.
        cells = np.ascontiguousarray(labels).astype(np.int32)
        if cells.shape != markers.shape[:2]:
            raise ValueError(f"mask is {cells.shape}, markers are {markers.shape[:2]}")

        cell_ids = np.unique(cells)
        cell_ids = cell_ids[cell_ids > 0]
        n_cells = cell_ids.size

        if n_cells == 0:
            for z in range(S):
                queue.put({"type": "channel", "z": z, "positive": 0.0,
                           "slice": np.zeros_like(cells, dtype=np.int16)})
            queue.put({"type": "done"})
            return

        # Every crop is taken whole, so an image smaller than the crop is padded.
        orig_h, orig_w = cells.shape
        pad_y = max(0, crop - orig_h)
        pad_x = max(0, crop - orig_w)
        if pad_y or pad_x:
            cells = np.pad(cells, ((0, pad_y), (0, pad_x)))
            markers = np.pad(markers, ((0, pad_y), (0, pad_x), (0, 0)))
        height, width = cells.shape

        centres = np.asarray(ndimage.center_of_mass(
            cells > 0, cells, cell_ids)).astype(np.int32)

        def make_sample(z, cell_id, centre_y, centre_x):
            """The same crop the model was trained on."""
            y = min(max(int(centre_y) - half, 0), height - crop)
            x = min(max(int(centre_x) - half, 0), width - crop)

            # Channels beyond either end of the stack do not exist and stay empty.
            wanted = np.arange(z - n_side, z + n_side + 1)
            exists = (wanted >= 0) & (wanted < S)

            planes = np.zeros((crop, crop, n_markers), dtype=np.float32)
            planes[..., exists] = markers[y:y + crop, x:x + crop, wanted[exists]]

            patch = cells[y:y + crop, x:x + crop]
            stack = [planes, (patch == cell_id)[..., None]]
            if use_other_cells:
                stack.append(((patch > 0) & (patch != cell_id))[..., None])

            return np.concatenate(stack, axis=-1).astype(np.float32)

        def predict(batch):
            """Probabilities for one batch of crops.

            Keras' predict() uploads each batch through a tf.data pipeline whose
            tensors it never lets go of, so calling it once per batch fills the
            card after a few dozen batches. Calling the model keeps one batch on
            the device at a time. A card smaller than the batch still asks for
            too much at once, so the batch is halved until it fits.
            """
            nonlocal batch_size
            while True:
                try:
                    return np.concatenate([
                        np.asarray(model(batch[i:i + batch_size], training=False)).ravel()
                        for i in range(0, len(batch), batch_size)])
                except tf.errors.ResourceExhaustedError:
                    if batch_size <= 16:
                        raise
                    batch_size //= 2
                    print(f"[AI] GPU out of memory, retrying with batch {batch_size}",
                          flush=True)

        call = np.zeros(int(cell_ids.max()) + 1, dtype=np.int16)

        # Fixed, so that predict() halving batch_size cannot move the staging
        # boundaries and leave cells out.
        chunk = batch_size

        for z in range(S):
            probability = np.empty(n_cells, dtype=np.float32)

            for start in range(0, n_cells, chunk):
                rows = np.arange(start, min(start + chunk, n_cells))
                batch = np.stack([make_sample(z, cell_ids[i], *centres[i])
                                  for i in rows])
                probability[rows] = predict(batch)

            # 0 stays background, 1 is negative and 2 is positive, as in the
            # training masks.
            call[:] = 0
            call[cell_ids] = np.where(probability >= threshold, 2, 1)

            # Sent as it is finished rather than all at the end, so the caller
            # can show progress and neither side has to hold every channel.
            queue.put({"type": "channel", "z": z,
                       "positive": float((probability >= threshold).mean()),
                       "slice": call[cells][:orig_h, :orig_w].copy()})

        queue.put({"type": "done"})
    except Exception as e:
        queue.put({"type": "error", "error": str(e), "traceback": traceback.format_exc()})
