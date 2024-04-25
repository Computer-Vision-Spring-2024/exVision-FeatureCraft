# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'task3_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1101, 732)
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(12)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.left_widget_corner_tab = QtWidgets.QWidget(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.left_widget_corner_tab.sizePolicy().hasHeightForWidth()
        )
        self.left_widget_corner_tab.setSizePolicy(sizePolicy)
        self.left_widget_corner_tab.setObjectName("left_widget_corner_tab")
        self.horizontalLayout_3.addWidget(self.left_widget_corner_tab)
        self.right_widget_corner_tab = QtWidgets.QWidget(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.right_widget_corner_tab.sizePolicy().hasHeightForWidth()
        )
        self.right_widget_corner_tab.setSizePolicy(sizePolicy)
        self.right_widget_corner_tab.setObjectName("right_widget_corner_tab")
        self.horizontalLayout_3.addWidget(self.right_widget_corner_tab)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.consumed_time_label = QtWidgets.QLabel(self.tab_2)
        self.consumed_time_label.setObjectName("consumed_time_label")
        self.horizontalLayout.addWidget(self.consumed_time_label)
        self.threshold_label = QtWidgets.QLabel(self.tab_2)
        self.threshold_label.setObjectName("threshold_label")
        self.horizontalLayout.addWidget(self.threshold_label)
        self.horizontalSlider_corner_tab = QtWidgets.QSlider(self.tab_2)
        self.horizontalSlider_corner_tab.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_corner_tab.setObjectName("horizontalSlider_corner_tab")
        self.horizontalLayout.addWidget(self.horizontalSlider_corner_tab)
        self.threshold_value_label = QtWidgets.QLabel(self.tab_2)
        self.threshold_value_label.setText("")
        self.threshold_value_label.setObjectName("threshold_value_label")
        self.horizontalLayout.addWidget(self.threshold_value_label)
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem)
        self.apply_harris_push_button = QtWidgets.QPushButton(self.tab_2)
        self.apply_harris_push_button.setObjectName("apply_harris_push_button")
        self.horizontalLayout.addWidget(self.apply_harris_push_button)
        self.apply_lambda_minus_push_button = QtWidgets.QPushButton(self.tab_2)
        self.apply_lambda_minus_push_button.setObjectName(
            "apply_lambda_minus_push_button"
        )
        self.horizontalLayout.addWidget(self.apply_lambda_minus_push_button)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.input_image_1_sift = QtWidgets.QFrame(self.tab)
        self.input_image_1_sift.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.input_image_1_sift.setFrameShadow(QtWidgets.QFrame.Raised)
        self.input_image_1_sift.setObjectName("input_image_1_sift")

        self.horizontalLayout_12.addWidget(self.input_image_1_sift)
        self.input_image_2_sift = QtWidgets.QFrame(self.tab)
        self.input_image_2_sift.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.input_image_2_sift.setFrameShadow(QtWidgets.QFrame.Raised)
        self.input_image_2_sift.setObjectName("input_image_2_sift")
        self.horizontalLayout_12.addWidget(self.input_image_2_sift)
        self.verticalLayout_4.addLayout(self.horizontalLayout_12)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.n_octaves = QtWidgets.QLabel(self.tab)
        self.n_octaves.setObjectName("n_octaves")
        self.horizontalLayout_8.addWidget(self.n_octaves)
        self.n_octaves_spin_box = QtWidgets.QSpinBox(self.tab)
        self.n_octaves_spin_box.setObjectName("n_octaves_spin_box")
        self.n_octaves_spin_box.setValue(4)
        self.n_octaves_spin_box.setSingleStep(1)
        self.n_octaves_spin_box.setMinimum(4)
        self.n_octaves_spin_box.setMaximum(8)
        self.horizontalLayout_8.addWidget(self.n_octaves_spin_box)
        self.horizontalLayout_10.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.s_value = QtWidgets.QLabel(self.tab)
        self.s_value.setObjectName("s_value")
        self.horizontalLayout_9.addWidget(self.s_value)
        self.s_value_spin_box = QtWidgets.QSpinBox(self.tab)
        self.s_value_spin_box.setObjectName("s_value_spin_box")
        self.s_value_spin_box.setValue(2)
        self.s_value_spin_box.setSingleStep(1)
        self.s_value_spin_box.setMinimum(2)
        self.s_value_spin_box.setMaximum(5)
        self.horizontalLayout_9.addWidget(self.s_value_spin_box)
        self.horizontalLayout_10.addLayout(self.horizontalLayout_9)
        self.verticalLayout_3.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.sigma_base = QtWidgets.QLabel(self.tab)
        self.sigma_base.setObjectName("sigma_base")
        self.horizontalLayout_7.addWidget(self.sigma_base)
        self.sigma_base_spin_box = QtWidgets.QDoubleSpinBox(self.tab)
        self.sigma_base_spin_box.setObjectName("sigma_base_spin_box")
        self.sigma_base_spin_box.setValue(1.6)
        self.sigma_base_spin_box.setSingleStep(0.1)
        self.sigma_base_spin_box.setMinimum(1.6)
        self.sigma_base_spin_box.setMaximum(3)
        self.horizontalLayout_7.addWidget(self.sigma_base_spin_box)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.r_ratio = QtWidgets.QLabel(self.tab)
        self.r_ratio.setObjectName("r_ratio")
        self.horizontalLayout_6.addWidget(self.r_ratio)
        self.r_ratio_spin_box = QtWidgets.QDoubleSpinBox(self.tab)
        self.r_ratio_spin_box.setObjectName("r_ratio_spin_box")
        self.r_ratio_spin_box.setValue(10)
        self.r_ratio_spin_box.setSingleStep(1)
        self.r_ratio_spin_box.setMinimum(6)
        self.r_ratio_spin_box.setMaximum(12)
        self.horizontalLayout_6.addWidget(self.r_ratio_spin_box)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_6)
        self.verticalLayout_3.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.contrast_th = QtWidgets.QLabel(self.tab)
        self.contrast_th.setObjectName("contrast_th")
        self.horizontalLayout_4.addWidget(self.contrast_th)
        self.contrast_th_slider = QtWidgets.QSlider(self.tab)
        self.contrast_th_slider.setOrientation(QtCore.Qt.Horizontal)
        self.contrast_th_slider.setObjectName("contrast_th_slider")
        self.contrast_th_slider.setValue(30)
        self.contrast_th_slider.setSingleStep(10)
        self.contrast_th_slider.setMinimum(1)
        self.contrast_th_slider.setMaximum(100)
        self.horizontalLayout_4.addWidget(self.contrast_th_slider)
        self.horizontalLayout_5.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.confusion_factor = QtWidgets.QLabel(self.tab)
        self.confusion_factor.setObjectName("confusion_factor")
        self.horizontalLayout_2.addWidget(self.confusion_factor)
        self.confusion_factor_slider = QtWidgets.QSlider(self.tab)
        self.confusion_factor_slider.setOrientation(QtCore.Qt.Horizontal)
        self.confusion_factor_slider.setObjectName("confusion_factor_slider")
        self.confusion_factor_slider.setValue(3)
        self.confusion_factor_slider.setSingleStep(1)
        self.confusion_factor_slider.setMinimum(1)
        self.confusion_factor_slider.setMaximum(10)
        self.horizontalLayout_2.addWidget(self.confusion_factor_slider)
        self.horizontalLayout_5.addLayout(self.horizontalLayout_2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.verticalLayout_4.addLayout(self.verticalLayout_3)
        self.sift_elapsed_time = QtWidgets.QLabel(self.tab)
        self.sift_elapsed_time.setMinimumSize(QtCore.QSize(0, 25))
        self.sift_elapsed_time.setMaximumSize(QtCore.QSize(16777215, 25))
        self.sift_elapsed_time.setObjectName("label")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_13.addItem(spacerItem1)
        self.apply_sift_vlayout = QtWidgets.QVBoxLayout()
        self.apply_sift_vlayout.setObjectName("apply_sift_vlayout")
        self.apply_sift = QtWidgets.QPushButton(self.tab)
        self.apply_sift.setObjectName("pushButton")
        self.apply_sift_vlayout.addWidget(self.apply_sift)
        self.apply_sift_vlayout.addWidget(self.sift_elapsed_time)
        self.horizontalLayout_13.addLayout(self.apply_sift_vlayout)
        spacerItem2 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_13.addItem(spacerItem2)
        self.verticalLayout_4.addLayout(self.horizontalLayout_13)
        self.tabWidget.addTab(self.tab, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout.setObjectName("gridLayout")
        self.output_image_sift = QtWidgets.QFrame(self.tab_3)
        self.output_image_sift.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.output_image_sift.setFrameShadow(QtWidgets.QFrame.Raised)
        self.output_image_sift.setObjectName("output_image_sift")
        self.gridLayout.addWidget(self.output_image_sift, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.region_growing_input = QtWidgets.QFrame(self.tab_4)
        self.region_growing_input.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.region_growing_input.setFrameShadow(QtWidgets.QFrame.Raised)
        self.region_growing_input.setObjectName("region_growing_input")
        self.horizontalLayout_14.addWidget(self.region_growing_input)
        self.region_growing_output = QtWidgets.QFrame(self.tab_4)
        self.region_growing_output.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.region_growing_output.setFrameShadow(QtWidgets.QFrame.Raised)
        self.region_growing_output.setObjectName("region_growing_output")
        self.horizontalLayout_14.addWidget(self.region_growing_output)
        self.gridLayout_2.addLayout(self.horizontalLayout_14, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_4, "")
        self.verticalLayout_2.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1101, 28))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_Image = QtWidgets.QAction(MainWindow)
        self.actionLoad_Image.setObjectName("actionLoad_Image")
        self.menuFile.addAction(self.actionLoad_Image)
        self.menubar.addAction(self.menuFile.menuAction())

        ## Harris Canvas
        self.harris_input_vlayout = QtWidgets.QVBoxLayout(self.left_widget_corner_tab)
        self.harris_input_vlayout.setObjectName("harris_input_hlayout")
        self.harris_input_figure = plt.figure()
        self.harris_input_figure_canvas = FigureCanvas(self.harris_input_figure)
        self.harris_input_vlayout.addWidget(self.harris_input_figure_canvas)
        ## Harris Output
        self.harris_output_vlayout = QtWidgets.QVBoxLayout(self.right_widget_corner_tab)
        self.harris_output_vlayout.setObjectName("harris_output_hlayout")
        self.harris_output_figure = plt.figure()
        self.harris_output_figure_canvas = FigureCanvas(self.harris_output_figure)
        self.harris_output_vlayout.addWidget(self.harris_output_figure_canvas)
        ## End of Harris canvas

        ## SIFT 1
        self.input_1_vlayout = QtWidgets.QVBoxLayout(self.input_image_1_sift)
        self.input_1_vlayout.setObjectName("input_1_hlayout")
        self.input_1_figure = plt.figure()
        self.input_1_figure_canvas = FigureCanvas(self.input_1_figure)
        self.input_1_vlayout.addWidget(self.input_1_figure_canvas)
        ## SIFT 2
        self.input_2_vlayout = QtWidgets.QVBoxLayout(self.input_image_2_sift)
        self.input_2_vlayout.setObjectName("input_2_hlayout")
        self.input_2_figure = plt.figure()
        self.input_2_figure_canvas = FigureCanvas(self.input_2_figure)
        self.input_2_vlayout.addWidget(self.input_2_figure_canvas)
        ## SIFT output
        self.output_vlayout = QtWidgets.QVBoxLayout(self.output_image_sift)
        self.output_vlayout.setObjectName("output_hlayout")
        self.output_figure = plt.figure()
        self.sift_output_figure_canvas = FigureCanvas(self.output_figure)
        self.output_vlayout.addWidget(self.sift_output_figure_canvas)
        ## End of SIFT canvas

        ## Region Growing Input
        self.region_growing_input_vlayout = QtWidgets.QVBoxLayout(
            self.region_growing_input
        )
        self.region_growing_input_vlayout.setObjectName("region_growing_input_hlayout")
        self.region_growing_input_figure = plt.figure()
        self.region_growing_input_figure_canvas = FigureCanvas(
            self.region_growing_input_figure
        )
        self.region_growing_input_vlayout.addWidget(
            self.region_growing_input_figure_canvas
        )
        ## Region Growing Output
        self.region_growing_output_vlayout = QtWidgets.QVBoxLayout(
            self.region_growing_output
        )
        self.region_growing_output_vlayout.setObjectName(
            "region_growing_output_hlayout"
        )
        self.region_growing_output_figure = plt.figure()
        self.region_growing_output_figure_canvas = FigureCanvas(
            self.region_growing_output_figure
        )
        self.region_growing_output_vlayout.addWidget(
            self.region_growing_output_figure_canvas
        )
        ## End of Region Growing canvas

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.consumed_time_label.setText(
            _translate("MainWindow", "This Operation consumed ..... seconds || ")
        )
        self.threshold_label.setText(_translate("MainWindow", "Adjust the threshold"))
        self.apply_harris_push_button.setText(_translate("MainWindow", "Apply Harris"))
        self.apply_lambda_minus_push_button.setText(
            _translate("MainWindow", "Apply lambda minus")
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_2),
            _translate("MainWindow", "Corner Detection"),
        )
        self.n_octaves.setText(_translate("MainWindow", "n_octaves: 4"))
        self.s_value.setText(_translate("MainWindow", "s_value: 2"))
        self.sigma_base.setText(_translate("MainWindow", "sigma_base: 1.6"))
        self.r_ratio.setText(_translate("MainWindow", "r_ratio: 10"))
        self.contrast_th.setText(_translate("MainWindow", "contrast_th: 0.03"))
        self.confusion_factor.setText(_translate("MainWindow", "confusion_factor: 0.3"))
        self.sift_elapsed_time.setText(_translate("MainWindow", "Elapsed Time is"))
        self.apply_sift.setText(_translate("MainWindow", "Apply SIFT"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab), _translate("MainWindow", "SIFT")
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "SIFT Output")
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_4),
            _translate("MainWindow", "Region Growing"),
        )
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionLoad_Image.setText(_translate("MainWindow", "Load Image"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
