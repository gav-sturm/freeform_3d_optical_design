import logging
import sys
from pathlib import Path

from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from qtpy.uic import loadUi

SRC_PATH = Path(__file__).parent
ROOT_PATH = SRC_PATH.parent
RESOURCES_PATH = ROOT_PATH / "gui"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi(RESOURCES_PATH / "main.ui", self)

def main():
    app = QApplication(sys.argv)

    # Create and show the main window.
    main_window = MainWindow()

    main_window.show()

    result = app.exec_()
    sys.exit(result)


if __name__ == "__main__":
    main()
