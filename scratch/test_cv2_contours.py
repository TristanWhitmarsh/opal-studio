import numpy as np
import cv2
import time

# Simulate a large mask (2000x2000) with 1000 cells
labels = np.zeros((2000, 2000), dtype=np.int32)
for i in range(1000):
    y, x = np.random.randint(20, 1980, size=2)
    labels[y-10:y+10, x-10:x+10] = i + 1

def test_cv2_contours(labels):
    start = time.perf_counter()
    all_contours = []
    unique_ids = np.unique(labels)
    for lid in unique_ids:
        if lid == 0: continue
        # Threshold for this label
        mask = (labels == lid).astype(np.uint8)
        # cv2.findContours is generally faster
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.append((lid, contours))
    end = time.perf_counter()
    print(f"cv2.findContours (per-label) took {end - start:.4f}s for {len(unique_ids)-1} cells")
    return all_contours

test_cv2_contours(labels)
