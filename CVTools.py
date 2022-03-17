import sys
import random
import time
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy import arange, sin, pi

class Menu(QMainWindow):
	'''
		Usage:
		>> bar = Menu()
		>> self.gridLayout.addWidget(bar,0,0)
	'''
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		# 新建一个菜单栏对象
		self.lbl = QtWidgets.QLabel('print to here',self)
		self.lbl.move(50,150)
		menubar = self.menuBar()
		# 添加“文件”菜单项
		fileMenu = menubar.addMenu('文件')
		# 添加“新建”菜单项
		newAct = QtWidgets.QAction('新建',self)
		# 添加“导入”菜单项 （带子菜单项）
		impMenu = QtWidgets.QMenu('导入',self)
		impAct1 = QtWidgets.QAction('从pdf导入',self)
		impAct2 = QtWidgets.QAction('从word导入', self)
		# 为菜单添加单击处理时间
		fileMenu.addAction(newAct)
		fileMenu.addMenu(impMenu)
		impMenu.addAction(impAct1)
		impMenu.addAction(impAct2)
		impAct1.triggered.connect(lambda :self.actionHandler(1))
		impAct2.triggered.connect(lambda :self.actionHandler(2))

		self.setGeometry(300,300,300,200)
		self.setWindowTitle('菜单')
		# self.show()

class Ui_MainWindow(object):	
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("MainWindow")
		self.screen = QtWidgets.QDesktopWidget().screenGeometry()
		MainWindow.setGeometry(100, 500, self.screen.width()/2,self.screen.height()/2)
		# MainWindow.resize(self.screen.width()/4,self.screen.height()/4)
		# MainWindow.showMaximized()
		self.centralWidget = QtWidgets.QWidget(MainWindow)
		self.centralWidget.setObjectName("centralWidget")
		MainWindow.setCentralWidget(self.centralWidget)
####################################################################################
		#QGridLayout
		self.gridLayout = QtWidgets.QGridLayout(self.centralWidget)
		# self.gridLayout.setContentsMargins(11, 11, 11, 11)
		self.gridLayout.setSpacing(6)
		self.gridLayout.setObjectName("gridLayout")
		#
		bar = Menu()
		self.gridLayout.addWidget(bar,0,0)
		
		#QVBoxLayout
		self.verticallayout = QtWidgets.QHBoxLayout()
		# self.verticallayout.setContentsMargins(11, 11, 11, 11)
		self.verticallayout.setSpacing(6)
		self.verticallayout.setObjectName("verticallayout")
		self.gridLayout.addLayout(self.verticallayout,1,0)
		
		#QGroupBox
		self.echoGroup =  QtWidgets.QGroupBox('Echo')
		self.echoLayout = QtWidgets.QGridLayout()
		self.echoGroup.setLayout(self.echoLayout)
		self.echoLabel = QtWidgets.QLabel('Mode:')
		self.echoLayout.addWidget(self.echoLabel,0,0)
		self.verticallayout.addWidget(self.echoGroup,0)
		
		#QLabel
		self.l1 = QtWidgets.QLabel('Yoking')
		self.l1.setObjectName('l1')
		self.verticallayout.addWidget(self.l1)
		#
		#QLabel
		self.l1 = QtWidgets.QLabel('Yoking')
		self.l1.setObjectName('l1')
		self.verticallayout.addWidget(self.l1)
		#
		#QLabel
		self.l1 = QtWidgets.QLabel('Yoking')
		self.l1.setObjectName('l1')
		self.verticallayout.addWidget(self.l1)
		#
		#LineEdit
		self.e1 = QtWidgets.QLineEdit()
		self.e1.setValidator(QtGui.QIntValidator())
		self.e1.setMaxLength(6)
		self.e1.setAlignment(QtCore.Qt.AlignRight)
		self.e1.setObjectName('e1')
		self.e1.setFont(QtGui.QFont("Arial",10))
		#
		self.gridLayout.addWidget(self.e1)
		#

		#
		



####################################################################################
if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	ex = Ui_MainWindow()
	w = QtWidgets.QMainWindow()
	ex.setupUi(w)
	w.show()
	sys.exit(app.exec_())
####################################################################################