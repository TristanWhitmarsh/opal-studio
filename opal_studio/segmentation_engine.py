import os
import sys
import traceback
import numpy as np

def run_segmentation_task_pipe(conn, params, input_channels_data):
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
            basedir = os.path.join(os.getcwd(), "models")
            
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
            from instanseg import InstanSeg
            import torch
            
            model_name = params["model_name"]
            local_model_dir = os.path.join(os.getcwd(), "models", "instanseg", model_name)
            if os.path.exists(os.path.join(local_model_dir, "instanseg.pt")):
                model_name = local_model_dir
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = InstanSeg(model_name, device=device)
            
            x = input_channels_data[0]
            x_input = x
            if "brightfield" in params["model_name"].lower() and x.ndim == 2:
                x_input = np.stack([x]*3, axis=-1)
            
            if max(x.shape[0], x.shape[1]) > 512:
                out, _ = model.eval_medium_image(x_input, params.get("pixel_size", 1.0), tile_size=512, batch_size=16)
            else:
                out, _ = model.eval_small_image(x_input, params.get("pixel_size", 1.0))
            
            labels_tensor = out[0]
            if hasattr(labels_tensor, 'cpu'):
                labels_tensor = labels_tensor.cpu().numpy()
            elif hasattr(labels_tensor, 'numpy'):
                labels_tensor = labels_tensor.numpy()
            
            labels_tensor = np.squeeze(labels_tensor)
            if labels_tensor.ndim == 3:
                # First channel is nuclei, rest are cells
                results.append((labels_tensor[0], "InstanSeg Nuclei", False))
                for c in range(1, labels_tensor.shape[0]):
                    results.append((labels_tensor[c], "InstanSeg Cells", True))
            else:
                results.append((labels_tensor, "InstanSeg", False))

        elif method == "mesmer":
            from deepcell.applications import Mesmer
            
            if params.get("api_key"):
                os.environ["DEEPCELL_ACCESS_TOKEN"] = params["api_key"]
            
            app = Mesmer()
            n_data = input_channels_data[0]
            if len(input_channels_data) > 1:
                m_data = input_channels_data[1]
            else:
                m_data = np.zeros_like(n_data)
                
            input_stack = np.stack([n_data, m_data], axis=-1)
            input_stack = np.expand_dims(input_stack, axis=0)
            
            labeled_combined = app.predict(input_stack, image_mpp=params.get("pixel_size", 1.0), batch_size=16, compartment=params.get("compartment", "nuclear"))
            
            if labeled_combined.shape[-1] >= 2:
                cell_labels = np.squeeze(labeled_combined[0, ..., 0]).astype(np.int32)
                nuc_labels = np.squeeze(labeled_combined[0, ..., 1]).astype(np.int32)
                results.append((nuc_labels, "Mesmer Nuclei", False))
                results.append((cell_labels, "Mesmer Cells", True))
            else:
                labels = np.squeeze(labeled_combined[0, ..., 0]).astype(np.int32)
                results.append((labels, "Mesmer", False))

        conn.send({"success": True, "results": results})
    except Exception as e:
        conn.send({"success": False, "error": str(e), "traceback": traceback.format_exc()})
    finally:
        conn.close()

def run_positivity_task(queue, params, data):
    """
    Worker function for AI cell positivity detection.
    """
    try:
        import tensorflow as tf
        from scipy.ndimage import label as cc_label
        
        labels = data["labels"]
        markers = data["markers"]
        
        model_path = os.path.join(os.getcwd(), "models", "cellpos", "marker_cnn_epoch_100.h5")
        model = tf.keras.models.load_model(model_path, compile=False)
        
        PATCH_SIZE = 64
        THRESHOLD = 0.5
        CHUNK_SIZE = 500
        
        S = markers.shape[2]
        results = []
        
        for z in range(S):
            img_curr = markers[:, :, z]
            # Simplified neighbor logic for worker
            img_prev = markers[:, :, max(0, z-1)]
            img_next = markers[:, :, min(S-1, z+1)]
            
            cell_region = labels > 0
            labeled_slice, num_components = cc_label(cell_region)
            
            if num_components == 0:
                results.append((np.zeros_like(labels, dtype=np.int16), z))
                continue
                
            slice_out = np.zeros_like(labels, dtype=np.int16)
            patches = []
            cell_ids = []
            
            # Identify cell centers and crop patches
            # This is a bit slow in Python, but isolating it helps with memory/DLLs
            for comp_id in range(1, num_components + 1):
                mask = (labeled_slice == comp_id)
                ys, xs = np.where(mask)
                if ys.size == 0: continue
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                
                y0, x0 = max(0, cy - 32), max(0, cx - 32)
                y1, x1 = min(labels.shape[0], y0 + 64), min(labels.shape[1], x0 + 64)
                # Ensure 64x64
                if y1 - y0 < 64: y0 = max(0, y1 - 64)
                if x1 - x0 < 64: x0 = max(0, x1 - 64)
                
                p_curr = img_curr[y0:y1, x0:x1]
                p_prev = img_prev[y0:y1, x0:x1]
                p_next = img_next[y0:y1, x0:x1]
                p_mask = mask[y0:y1, x0:x1].astype(np.float32)
                
                stacked = np.stack([p_prev, p_curr, p_next, p_mask], axis=-1)
                patches.append(stacked)
                cell_ids.append(comp_id)
                
                if len(patches) >= CHUNK_SIZE:
                    X = np.stack(patches, axis=0)
                    probs = model.predict(X, batch_size=512, verbose=0).reshape(-1)
                    for cid, p in zip(cell_ids, probs):
                        slice_out[labeled_slice == cid] = 2 if p >= THRESHOLD else 1
                    patches.clear()
                    cell_ids.clear()
            
            if patches:
                X = np.stack(patches, axis=0)
                probs = model.predict(X, batch_size=512, verbose=0).reshape(-1)
                for cid, p in zip(cell_ids, probs):
                    slice_out[labeled_slice == cid] = 2 if p >= THRESHOLD else 1
                    
            results.append((slice_out, z))
            
        queue.put({"success": True, "results": results})
    except Exception as e:
        queue.put({"success": False, "error": str(e), "traceback": traceback.format_exc()})
