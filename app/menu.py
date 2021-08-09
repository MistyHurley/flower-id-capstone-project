from PyQt5.QtWidgets import*
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from utility import *
import matplotlib
matplotlib.use('Qt5Agg')


class Menu(QWidget):
    filename = ''

    flowers_scanned_total = 0
    flowers_scanned = [0, 0, 0, 0, 0]
    average_confidence = 0
    confidence_history = []

    def __init__(self, parent=None):
        super(Menu, self).__init__(parent)

        self.setWindowTitle("Downtown Botanical Garden Flower ID")

        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        main_menu_tab = QWidget()
        confidence_history_tab = QWidget()
        statistics_tab = QWidget()
        tabs.addTab(main_menu_tab, "Main Menu")
        tabs.addTab(confidence_history_tab, "Confidence History")
        tabs.addTab(statistics_tab, "Statistics")

        layout.addWidget(tabs)

        # main menu tab

        main_menu_tab.layout = QVBoxLayout(self)

        welcome = QLabel("Welcome to the Downtown Botanical Garden flower identification tool!")
        boldfont = QFont()
        boldfont.setBold(True)
        welcome.setFont(boldfont)
        welcome.setAlignment(Qt.AlignHCenter)
        main_menu_tab.layout.addWidget(welcome)
        instructions = QLabel("Select an image using the button below for quick and easy identification.")
        instructions.setAlignment(Qt.AlignHCenter)
        main_menu_tab.layout.addWidget(instructions)

        self.open_button = QPushButton("Open...")
        self.open_button.setMinimumSize(200, 50)
        self.open_button.clicked.connect(self.open_file)
        main_menu_tab.layout.addWidget(self.open_button)

        main_menu_hbox = QHBoxLayout()

        selected_image_vbox = QVBoxLayout()

        self.selected_image_label = QLabel()
        self.selected_image_label.setAlignment(Qt.AlignCenter)
        self.selected_image_label.setFixedSize(300, 200)
        selected_image_vbox.addWidget(self.selected_image_label)

        self.selected_image_prediction = QLabel()
        self.selected_image_prediction.setAlignment(Qt.AlignCenter)
        selected_image_vbox.addWidget(self.selected_image_prediction)

        main_menu_hbox.addLayout(selected_image_vbox)

        self.likelihood_plot = GraphWidget(self, width=5, height=2, dpi=70)
        self.likelihood_plot.axes.set_ylim(ymin=0, ymax=100)
        self.likelihood_plot.axes.set_ylabel('Relative Confidence')
        self.likelihood_plot.axes.bar(class_names, [0, 0, 0, 0, 0])
        main_menu_hbox.addWidget(self.likelihood_plot)

        main_menu_tab.layout.addLayout(main_menu_hbox)

        main_menu_tab.setLayout(main_menu_tab.layout)

        # confidence history tab

        confidence_history_tab.layout = QVBoxLayout(self)

        self.confidence_history_header = QLabel("Confidence of 0 last scans (oldest to newest):")
        self.confidence_history_header.setAlignment(Qt.AlignHCenter)
        confidence_history_tab.layout.addWidget(self.confidence_history_header)

        self.confidence_history_plot = GraphWidget(self, width=10, height=4, dpi=70)
        self.confidence_history_plot.axes.set_ylim(ymin=-2, ymax=102)
        self.confidence_history_plot.axes.set_xlim(xmin=-0.5, xmax=9.5)
        self.confidence_history_plot.axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.confidence_history_plot.axes.xaxis.set_visible(False)
        self.confidence_history_plot.axes.set_ylabel('Confidence')
        self.confidence_history_plot.axes.plot(range(0, len(self.confidence_history)), self.confidence_history, 'o-b')
        confidence_history_tab.layout.addWidget(self.confidence_history_plot)

        self.confidence_average_label = QLabel("Session Average Confidence: N/A")
        self.confidence_average_label.setAlignment(Qt.AlignHCenter)
        confidence_history_tab.layout.addWidget(self.confidence_average_label)

        confidence_history_tab.setLayout(confidence_history_tab.layout)

        # statistics tab

        statistics_tab.layout = QVBoxLayout(self)

        self.statistics_plot = GraphWidget(self, width=10, height=4, dpi=70)
        self.statistics_plot.axes.set_ylim(ymin=0)
        self.statistics_plot.axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.statistics_plot.axes.set_ylabel('Quantity Identified')
        self.statistics_plot.axes.bar(class_names, [0, 0, 0, 0, 0])
        statistics_tab.layout.addWidget(self.statistics_plot)

        statistics_tab.setLayout(statistics_tab.layout)

        self.setLayout(layout)

    def open_file(self):
        self.filename = QFileDialog.getOpenFileName(self, "Open...", ".", "Image files (*.jpg)")
        print(self.filename)
        if len(self.filename[0]):
            pixmap = QPixmap(self.filename[0]).scaled(self.selected_image_label.size(),
                                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.selected_image_label.setPixmap(pixmap)
            predictions = Utility.get_model_prediction(self.filename[0])
            score = tf.nn.softmax(predictions[0])

            plot_scores = []
            for prediction in predictions[0]:
                plot_scores.append(100 * np.max(prediction))
            #print(plot_scores)

            #confidence = 100 * np.max(score)
            confidence = plot_scores[np.argmax(score)]
            prediction_text = "{} ({:.2f}% confident)".format(class_names[np.argmax(score)], confidence)
            self.selected_image_prediction.setText(prediction_text)

            self.likelihood_plot.axes.clear()
            self.likelihood_plot.axes.set_ylim(ymin=0, ymax=100)
            self.likelihood_plot.axes.set_ylabel('Relative Confidence')
            self.likelihood_plot.axes.bar(class_names, plot_scores)
            self.likelihood_plot.redraw()

            self.flowers_scanned_total += 1
            self.flowers_scanned[np.argmax(score)] += 1
            self.confidence_history.append(confidence)
            if len(self.confidence_history) > 10:
                self.confidence_history.pop(0)
            self.average_confidence = \
                (self.flowers_scanned_total - 1) / self.flowers_scanned_total * self.average_confidence + \
                1 / self.flowers_scanned_total * confidence

            #print(self.flowers_scanned_total)
            #print(self.flowers_scanned)
            #print(self.confidence_history)
            #print(self.average_confidence)

            self.update_session_data()

    def update_session_data(self):
        self.statistics_plot.axes.clear()
        self.statistics_plot.axes.set_ylim(ymin=0, ymax=np.max(self.flowers_scanned))
        self.statistics_plot.axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.statistics_plot.axes.set_ylabel('Quantity Identified')
        self.statistics_plot.axes.bar(class_names, self.flowers_scanned)
        self.statistics_plot.redraw()

        self.confidence_history_plot.axes.clear()
        self.confidence_history_plot.axes.set_ylim(ymin=-2, ymax=102)
        self.confidence_history_plot.axes.set_xlim(xmin=-0.5 - (10 - len(self.confidence_history)), xmax=9.5 - (10 - len(self.confidence_history)))
        self.confidence_history_plot.axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.confidence_history_plot.axes.xaxis.set_visible(False)
        self.confidence_history_plot.axes.set_ylabel('Confidence')
        self.confidence_history_plot.axes.plot(range(0, len(self.confidence_history)), self.confidence_history, 'o-b')
        self.confidence_history_plot.redraw()

        self.confidence_history_header.setText("Confidence of {} last scans (oldest to newest):".format(len(self.confidence_history)))
        self.confidence_average_label.setText("Session Average Confidence: {:.2f}%".format(self.average_confidence))


class GraphWidget(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(1, 1, 1)
        super(GraphWidget, self).__init__(self.fig)

    def redraw(self):
        self.fig.canvas.draw()
