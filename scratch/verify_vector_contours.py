import sys
import numpy as np
from PySide6.QtWidgets import QApplication
from opal_studio.main_window import MainWindow

def verify_vector_contours():
    # Create labels
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[10:30, 10:30] = 1 # Square
    
    app = QApplication(sys.argv)
    win = MainWindow()
    
    contour_data = win._get_contour_data(labels)
    
    # Check structure
    assert 1 in contour_data
    assert "polygons" in contour_data[1]
    assert "bbox" in contour_data[1]
    
    polygons = contour_data[1]["polygons"]
    assert len(polygons) == 1
    poly = polygons[0]
    
    # Square 20x20. find_contours with level=0.5 on a binary mask
    # should return points roughly along the boundary.
    print(f"Contour points count: {poly.count()}")
    
    # Bbox should be [y0, x1, y1, x1] -> [y0, x0, y1, x1]
    # Wait, my bbox was [loc[0].start, loc[1].start, loc[0].stop, loc[1].stop]
    assert contour_data[1]["bbox"] == [10, 10, 30, 30]
    
    # Verify coordinates are in image space + 0.5 shift
    # Top edge was 10.0, with shift it should be 10.5
    brect = poly.boundingRect()
    print(f"Bounding Rect: {brect}")
    
    assert abs(brect.top() - 10.5) < 0.1
    assert abs(brect.left() - 10.5) < 0.1
    assert abs(brect.bottom() - 30.5) < 0.1
    assert abs(brect.right() - 30.5) < 0.1
    
    print("Vector contour extraction PASSED")

if __name__ == "__main__":
    verify_vector_contours()
