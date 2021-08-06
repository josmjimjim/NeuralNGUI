import sys, os
from PyQt5.QtWidgets import  (QApplication, QWidget, QListWidget,
    QVBoxLayout, QListWidgetItem, QGridLayout, QGroupBox, QLineEdit, QLabel,
    QPushButton, QMessageBox, QFileDialog, QCheckBox)
from PyQt5.QtCore import QSize, Qt, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QPainter, QColor

class CentralWidget(QWidget):

    __slots__ = ['drag', 'file', 'pretrained', ]

    accept_signal = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create url signal
        self.pretrained = True

        # Create instances of object to call in initializeUI
        self.drag = DragandDropFiles(self)
        self.file = FileDirectorySystemBar(self)

        # Conect the signal and slot
        self.drag.file_directory.connect(self.file.recibeData)
        self.file.file_directory.connect(self.drag.updateIcon)

        # Initialize and display window
        self.initializeUI()

    def initializeUI(self):
        # Create vertical main layout
        v_box = QVBoxLayout()

        # Create group for the file directory drop
        group = QGroupBox("Drop the weights here", self)
        layout = QVBoxLayout(group)
        layout.addWidget(self.file)
        layout.addWidget(self.drag)
        group.setLayout(layout)

        # ADD widget to main layout
        v_box.addWidget(group)
        self.setLayout(v_box)

class DragandDropFiles(QListWidget):
    __slots__ = ['file_path', ]
    file_directory = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_path = None
        self.setAcceptDrops(True)
        self.setViewMode(QListWidget.IconMode)

    def dragEnterEvent(self, event):
        data = event.mimeData()
        if data.hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        data = event.mimeData()
        if data.hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        data = event.mimeData()
        # Check if data has urls and it is local url
        if data.hasUrls and data.urls() and data.urls()[0].isLocalFile():
            url = data.urls()[0].toLocalFile()
            # Check if it is the correct extension and display icon
            if os.path.splitext(url)[1].lower() == '.pth':
                event.setDropAction(Qt.CopyAction)
                event.accept()
                self.file_path = url
                self.clear()
                self.displayIcons()
                # Emit signal to display url text in file directory system
                self.file_directory.emit(url)
            else:
                QMessageBox().warning(self, "",
                                        "Error, the file extension is not valid",
                                        QMessageBox.Ok, QMessageBox.Ok)
                event.ignore()
        else:
            event.ignore()

    def displayIcons(self):
        # Set icon size and description text
        self.setIconSize(QSize(50, 50))
        icon = QListWidgetItem()
        icon.setText(self.file_path.split("/")[-1])
        file = os.getcwd()
        file = os.path.join(file,'assets/file.svg')
        icon.setIcon(QIcon(file))
        self.addItem(icon)

    @pyqtSlot()
    def paintEvent(self, event):
        # Paint event to set up background help text
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        painter.setPen(QColor(171, 178, 185))
        painter.setFont(QFont('Helvetica', 14))
        painter.drawText(self.rect(), Qt.AlignCenter, 'Please drop '+
                         'weight file here!')

    @pyqtSlot(str)
    def updateIcon(self, url):
        if os.path.splitext(url)[1].lower() == '.pth':
            self.file_path = url
            self.displayIcons()
        else:
            pass

class FileDirectorySystemBar(QWidget):
    __slots__ = ['file', 'directory', ]

    file_directory = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.displayFileBox()

    def displayFileBox(self):
        # Create group to the file directory
        group = QGroupBox('Weights File directory')

        # Create the text box to append file path and button directory
        self.file = QLineEdit(self)

        # Create push button to open directory finder
        dir_button = QPushButton('...')
        dir_button.setToolTip("Select weights file directory.")
        dir_button.clicked.connect(self.setDirectory)

        # Create layout for group
        layout = QVBoxLayout()

        # Organize widget in grid layout
        grid = QGridLayout()
        grid.addWidget(self.file, 0, 0)
        grid.addWidget(dir_button, 0, 1)

        # Add grid layout  and label to group layout
        layout.addLayout(grid)
        layout.addStretch()

        # Set layouts to groups
        group.setLayout(layout)

        # Widget Layout
        optionsLayout = QVBoxLayout()
        optionsLayout.addWidget(group)
        self.setLayout(optionsLayout)

    @pyqtSlot(str)
    def recibeData(self, data):
        # Slot for recibing data from drag and drop
        self.file.setText(data)

    def setDirectory(self):
        # Display file dialog
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.Directory)
        # Display directory in open mode
        self.directory, _ = file_dialog.getOpenFileName(self, 'Open File', '~',
                                                            "All Files (*);;")
        # Check file extension
        if self.directory:
            if os.path.splitext(self.directory)[1].lower() == '.pth':
                    self.file.setText(self.directory)
                    self.file_directory.emit(self.directory)
            else:
                QMessageBox().warning(self, "",
                        "Error, the file cannot be open because its extension is not valid",
                        QMessageBox.Ok, QMessageBox.Ok)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CentralWidget()
    window.show()
    sys.exit(app.exec_())

