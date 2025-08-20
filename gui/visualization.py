from typing import Optional
from qtpy.QtWidgets import QWidget, QLabel, QSizePolicy, QVBoxLayout
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from visdata import VisData


class Visualization(QWidget):
    """Abstract Visualization corresponding to a Qt widget.

    Subclasses must define:
    - name: a human-readable name (string)
    - update(self, data: VisData): called whenever new data is available
    """

    name: str = "Unnamed Visualization"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

    def update(self, data: VisData):  # noqa: D401 - method intentionally named per spec
        """Receive updates to the underlying VisData. Must be implemented by subclasses."""
        raise NotImplementedError

class LossHistoryVisualization(Visualization):
    name = "Loss History"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image_label = QLabel("No loss_history.png yet.")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setScaledContents(True)

        layout = QVBoxLayout()
        layout.addWidget(self._image_label)
        self.setLayout(layout)

    def update(self, data: VisData):
        img = data.loss_history_image
        if img is None:
            self._image_label.setText("loss_history.png not found.")
            return
        # Convert QImage -> QPixmap and display
        pix = QPixmap.fromImage(img)
        self._image_label.setPixmap(pix)
        self._image_label.setMinimumSize(320, 240)
