import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction


class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        # Call main initialize method to display contents
        self.initializeUI()

    def initializeUI(self):
        # Set window title
        self.setWindowTitle('Neuroscience Lab')
        # Set background as central widget
        # Display menu bar and MainWindow with show method
        self.displayMenuBar()
        self.show()

    def displayMenuBar(self):
        # Create actions for file menu
        self.open_act = QAction("Open", self)
        self.open_act.setShortcut('Ctrl+O')
        self.open_act.setStatusTip('Open a new file')
        self.open_act.triggered.connect(self.openFile)

        self.exit_act = QAction('Exit', self)
        self.exit_act.setShortcut('Ctrl+Q')
        self.open_act.setStatusTip('Quit program')
        self.exit_act.triggered.connect(self.closeWindows)

        # Create menubar
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)

        # Create file menu and add actions
        file_menu = menu_bar.addMenu('File')
        file_menu.addAction(self.open_act)
        file_menu.addSeparator()
        file_menu.addSeparator()
        file_menu.addAction(self.exit_act)

    def closeWindows(self):
        # Close the window when button is clicked
        self.close()

    def openFile(self):
        '''
        Open the window file directory when button is clicked
        '''



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())