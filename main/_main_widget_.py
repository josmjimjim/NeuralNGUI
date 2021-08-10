import sys, os
from ctypes import cdll
from PyQt5.QtWidgets import (QApplication, QWidget, QListWidget,
    QVBoxLayout, QListWidgetItem, QGridLayout, QGroupBox, QLineEdit, QLabel, QHBoxLayout,
    QPushButton, QMessageBox, QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox,
    QPlainTextEdit, QCheckBox)

from PyQt5.QtCore import QSize, Qt, QProcess, QProcessEnvironment, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QPainter, QColor
from stylesheet import __dict_style__


class Button(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(__dict_style__[text.lower()])

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
        self.setStyleSheet(__dict_style__['drop_style'])

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(1, 10000)

class DefineLearning(QDoubleSpinBox):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(0.0, 100.0)
        self.setSingleStep(0.001)
        self.setDecimals(6)
        self.setValue(0.001)

class LogsOutProcess(QPlainTextEdit):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

    @pyqtSlot(str)
    def messagesOUT(self, s):
        self.appendPlainText(s)

class FileDirectorySelect(QWidget):

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.directory = None
        self.kind = kind
        self.displayFileBox()

    def displayFileBox(self):
        if self.kind == 's':
            msg = 'Select File directory to save model'
        elif self.kind == 'train':
            msg = 'Select Train directory'
        elif self.kind == 'test':
            msg = '(Optional) Select Test directory'
        else:
            msg = 'Select File directory'
        # Create group to the file directory
        group = QGroupBox(msg)

        # Create the text box to append file path and button directory
        self.file = QLineEdit()

        # Create push button to open directory finder
        dir_button = QPushButton('...')
        dir_button.setToolTip("Select the file directory.")
        dir_button.clicked.connect(self.setSaveDirectory)

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

    def setSaveDirectory(self):
        # Display file dialog
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.Directory)
        # Display directory in open mode
        self.directory = file_dialog.getExistingDirectory(self, "Select Directory")
        # Check file extension
        if self.directory:
            self.file.setText(self.directory)
        else:
            self.file.clear()
            QMessageBox().warning(self, "",
                "Error, the file directory is empty!",
                QMessageBox.Ok, QMessageBox.Ok)

class ExternalProcess(QProcess):

    def __init__(self, parent=None):
        super().__init__(parent)
        env = self.read_env()
        self.setProcessEnvironment(env)
        self.setProcessChannelMode(QProcess.MergedChannels)

        # Check codec if window try and mac os, unix except
        try:
            self.codec = 'cp' + str(cdll.kernel32.GetACP())
        except:
            self.codec = 'utf-8'

    @staticmethod
    def read_env():
        dir_path = os.getcwd()
        dir_path = os.path.normpath(os.path.join(dir_path,
                                 '/main/assets/venv.txt'))
        env = QProcessEnvironment.systemEnvironment()
        try:
            with open(dir_path + '/main/assets/venv.txt', 'r') as f:
                venv_dir = f.read()
                f.close()
        except Exception:
               venv_dir = None

        env.insert("PYTHONPATH", venv_dir)

        return env

class CentralWidget(QWidget):

    file: object = None

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

        # Initialize widgets
        self.train = FileDirectorySelect('train', self)
        self.train.setToolTip('Select train directory')

        self.test = FileDirectorySelect('test', self)
        self.test.setToolTip('Select test directory')

        self.model = SelectOptions(self.__model_list, self)
        self.model.setToolTip('Select Neural Network Model')

        self.optimizer = SelectOptions(self.__optim_list, self)
        self.optimizer.setToolTip('Select Optimizer')

        self.batch = DefineBESize(self)
        self.batch.setToolTip('Select Batch Size')

        self.epochs = DefineBESize(self)
        self.epochs.setToolTip('Select Epochs for Training')

        self.lr = DefineLearning(self)
        self.lr.setToolTip('Select Learning Rate for Training')

        self.save = FileDirectorySelect('s', self)

        self.logs = LogsOutProcess(self)

        self.accept = Button('Accept', self)
        self.cancel = Button('Cancel', self)


        # Set up buttons actions / connections
        self.accept.clicked.connect(self.acceptAction)
        self.cancel.clicked.connect(self.cancelAction)

        # Define process
        self.process = None
        self.file = None

        # Initialize and display window
        self.initializeUI()

    def initializeUI(self):
        # Create vertical/ horizontal main layout
        v_box = QVBoxLayout()
        h_box = QHBoxLayout()

        # Create group for model properties
        g_dataset = QGroupBox("Model Dataset Definitions")
        l_dataset = QGridLayout()
        l_dataset.addWidget(self.train, 0, 0)
        l_dataset.addWidget(self.test, 1, 0)
        g_dataset.setLayout(l_dataset)

        # Create group for model properties
        g_model = QGroupBox("Model Properties Definitions")
        l_model = QGridLayout()
        label_mo = QLabel('Model / Optimizer')
        l_model.addWidget(label_mo, 0, 0)
        l_model.addWidget(self.model, 1, 0)
        l_model.addWidget(self.optimizer, 1, 1)
        label_be = QLabel('Batch Size / Epochs / Learning rate')
        l_model.addWidget(label_be, 2, 0)
        l_model.addWidget(self.batch, 3, 0)
        l_model.addWidget(self.epochs, 3, 1)
        l_model.addWidget(self.lr, 3, 2)
        label_pre = QLabel('Pretrained?')
        l_model.addWidget(self.check_button, 3, 3)
        l_model.addWidget(label_pre, 3, 4)
        g_model.setLayout(l_model)

        # Create group for the file directory and drop
        self.g_weights = QGroupBox("Weights Finder (Optional)")

        # Train model group
        right_box = QVBoxLayout()
        right_box.addWidget(self.logs)
        h_box_button = QHBoxLayout()
        h_box_button.addWidget(self.accept)
        h_box_button.addWidget(self.cancel)
        right_box.addLayout(h_box_button)

        # ADD widget to main layout
        v_box.addWidget(g_dataset)
        v_box.addWidget(g_model)
        v_box.addWidget(self.save)
        v_box.addWidget(self.g_weights)
        h_box.addLayout(v_box)
        h_box.addLayout(right_box)

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

            l_weights = QVBoxLayout()
            l_weights.addWidget(self.file)
            l_weights.addWidget(self.drag)
            self.g_weights.setLayout(l_weights)

        else:
            for widget in self.g_weights.children():
                widget.deleteLater()

    def acceptAction(self):

        # Clear logs windows
        self.logs.clear()
        self.logs.appendPlainText('Process starts running')

        file = os.getcwd()
        file = os. path.normpath(os.path.join(file,
                                'neuralnetwork.py'))

        process_args = [file]

        # Order arguments
        __args = {
            'model': self.model.currentText(),
            'optimizer': self.optimizer.currentText(),
            'batch': self.batch.value(),
            'epochs': self.epochs.value(),
            'lr': self.lr.value(),
            'save': self.save.directory,
            'train': self.train.directory,
        }

        for key in __args.keys():
            process_args.append(str(__args[key]))

        if self.test.directory:
            process_args.append('-t ' + str(self.test.file.text()))
        if self.file:
            process_args.append('-w ' + str(self.file.file.text()))

        if not self.process:
            self.process = ExternalProcess(self)
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stderr)
            self.process.finished.connect(self.process_finished)  # Clean up once complete.
            self.process.setProgram('python')
            self.process.setArguments(process_args)
            self.process.start()

    def cancelAction(self):
        if self.process:
            self.process.kill()
            self.logs.appendPlainText('Cancelled by user')
        else:
            pass

    def message(self, *args):
        self.logs.appendPlainText(*args)

    def process_finished(self):
        self.logs.appendPlainText('Process finished')
        self.process = None

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode(self.process.codec)
        self.message(stderr)

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode(self.process.codec)
        self.message(stdout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CentralWidget()
    window.show()
    sys.exit(app.exec_())

