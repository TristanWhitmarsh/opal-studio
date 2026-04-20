import sys
import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor
from opal_studio.channel_model import Channel, ChannelListModel
from opal_studio.main_window import MainWindow

def test_contour_generation():
    # Create labels
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[10:30, 10:30] = 1
    labels[50:70, 50:70] = 2
    
    # We need a MainWindow instance to use its _get_contour_data helper
    app = QApplication(sys.argv)
    win = MainWindow()
    
    contours = win._get_contour_data(labels)
    
    # Check if contours are generated
    # Boundary of a 20x20 square should have non-zero pixels
    # mode='inner' means it's the perimeter pixels of the objects
    assert np.any(contours == 1)
    assert np.any(contours == 2)
    
    # Check that it's just the boundary
    # The center of the squares should be 0 in contours
    assert contours[20, 20] == 0
    assert contours[60, 60] == 0
    
    # Check that it matches the labels
    assert contours[10, 10] == 1
    assert contours[50, 50] == 2
    
    print("Contour generation test PASSED")
    
if __name__ == "__main__":
    test_contour_generation()
