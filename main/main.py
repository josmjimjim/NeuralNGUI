import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget


class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        # Call main initialize method to display contents
        self.initializeUI()

    def initializeUI(self):
        # Set window title
        self.setWindowTitle('Neuroscience Lab')
        # Display menu bar and MainWindow with show method
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())