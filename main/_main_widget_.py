import sys, os
from PyQt5.QtWidgets import (QApplication, QWidget, QListWidget,
    QVBoxLayout, QListWidgetItem, QGridLayout, QGroupBox, QLineEdit, QLabel, QHBoxLayout,
    QPushButton, QMessageBox, QFileDialog, QComboBox, QSpinBox, QPlainTextEdit, QCheckBox)
from PyQt5.QtCore import QSize, Qt, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QPainter, QColor


class CheckPreTrained(QCheckBox):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEnabled(True)

class DragandDropFiles(QListWidget):

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
        file = os.path.join(file, 'assets/file.svg')
        icon.setIcon(QIcon(file))
        self.addItem(icon)

    @pyqtSlot()
    def paintEvent(self, event):
        # Paint event to set up background help text
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        painter.setPen(QColor(171, 178, 185))
        painter.setFont(QFont('Helvetica', 14))
        painter.drawText(self.rect(), Qt.AlignCenter, 'Please drop ' +
                         'weight file here!')

    @pyqtSlot(str)
    def updateIcon(self, url):
        if os.path.splitext(url)[1].lower() == '.pth':
            self.file_path = url
            self.displayIcons()
        else:
            pass

class FileDirectorySystemBar(QWidget):

    file_directory = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.displayFileBox()

    def displayFileBox(self):
        # Create group to the file directory
        group = QGroupBox('Weights File directory')

        # Create the text box to append file path and button directory
        self.file = QLineEdit()

        # Create push button to open directory finder
        dir_button = QPushButton('...')
        dir_button.setToolTip("Select weights file directory.")
        dir_button.clicked.connect(self.setDirectory)

        # Create layout for group
        layout = QHBoxLayout()
        layout.addWidget(self.file)
        layout.addWidget(dir_button)

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

class SelectOptions(QComboBox):

    def __init__(self, opt_list, parent=None):
        super().__init__(parent)
        for items in opt_list:
            self.addItem(items)

class DefineBESize(QSpinBox):

    be_size = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(1, 10000)
        self.valueChanged.connect(self.sendBESize)

    def sendBESize(self):
        self.be_size.emit(self.value())

class LogsOutProcess(QPlainTextEdit):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

    @pyqtSlot(str)
    def messagesOUT(self, s):
        self.appendPlainText(s)

class CentralWidget(QWidget):

    __model_list = ('resnet18', 'inceptionv3','googlenet',
                    'resnet34', 'resnet152', 'wideresnet50',
                    'alexnet', 'vgg16', 'mobilenetv2',
                    'mobilenet_v3_large','mobilenetv3small',
                    'mnasnet', 'vgg19',)
    __optim_list = ('Adam', 'AdamW', 'SGD', 'LBFGS',
                    'SparseAdam', 'RMSprop',)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create instances of object to call in initializeUI
        self.check_button = CheckPreTrained(self)
        self.check_button.stateChanged.connect(self.updateUI)
        self.model = SelectOptions(self.__model_list, self)
        self.model.setToolTip('Select Neural Network Model')
        self.optimizer = SelectOptions(self.__optim_list, self)
        self.optimizer.setToolTip('Select Optimizer')
        self.batch = DefineBESize(self)
        self.batch.setToolTip('Select Batch Size')
        self.epochs = DefineBESize(self)
        self.epochs.setToolTip('Select Epochs for Training')
        self.logs = LogsOutProcess(self)

        # Initialize and display window
        self.initializeUI()

    def initializeUI(self):
        # Create vertical/ horizontal main layout
        self.v_box = QVBoxLayout()
        h_box = QHBoxLayout()

        # Create group for model properties
        g_model = QGroupBox("Model Properties Definitions")
        l_model = QGridLayout()
        label_mo = QLabel('Model / Optimizer')
        l_model.addWidget(label_mo, 0, 0)
        l_model.addWidget(self.model, 1, 0)
        l_model.addWidget(self.optimizer, 1, 1)
        label_be = QLabel('Batch Size / Epochs')
        l_model.addWidget(label_be, 2, 0)
        l_model.addWidget(self.batch, 3, 0)
        l_model.addWidget(self.epochs, 3, 1)
        label_pre = QLabel('Pretrained?')
        l_model.addWidget(self.check_button, 3, 2)
        l_model.addWidget(label_pre, 3, 3)
        g_model.setLayout(l_model)

        # ADD widget to main layout
        self.v_box.addWidget(g_model)
        h_box.addLayout(self.v_box)
        h_box.addWidget(self.logs)

        self.setLayout(h_box)

    def updateUI(self):
        # Set check True
        if self.check_button.isChecked() == True:

            # Initialize widgets
            self.drag = DragandDropFiles(self)
            self.file = FileDirectorySystemBar(self)

            # Conect the signal and slot
            self.drag.file_directory.connect(self.file.recibeData)
            self.file.file_directory.connect(self.drag.updateIcon)

            # Create group for the file directory and drop
            g_weights = QGroupBox("Weights Finder (Optional)")
            l_weights = QVBoxLayout()
            l_weights.addWidget(self.file)
            l_weights.addWidget(self.drag)
            g_weights.setLayout(l_weights)
            # ADD widget to main layout
            self.v_box.addWidget(g_weights)

        else:
            self.v_box.itemAt(1).widget().deleteLater()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CentralWidget()
    window.show()
    sys.exit(app.exec_())

