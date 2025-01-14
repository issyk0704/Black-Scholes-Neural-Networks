from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from selection_screen import SelectionScreen
from black_scholes_Widget import BlackScholesWidget
from neural_network_widget import NeuralNetworkWidget
from OptionsDataWidget import OptionsDataWidget


class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Financial Analysis Application")
        self.stackedWidget = QStackedWidget(self)
        self.setCentralWidget(self.stackedWidget)

        # Initialize screens
        self.selectionScreen = SelectionScreen(self.switchToBlackScholes, self.switchToNeuralNetwork, self.switchToOptionsWidget)
        self.blackScholesWidget = BlackScholesWidget(self.switchToSelectionScreen)
        self.neuralNetworkWidget = NeuralNetworkWidget(self.switchToSelectionScreen)
        self.OptionsDataWidget = OptionsDataWidget(self.switchToSelectionScreen)

        # Add screens to the stacked widget
        self.stackedWidget.addWidget(self.selectionScreen)
        self.stackedWidget.addWidget(self.blackScholesWidget)
        self.stackedWidget.addWidget(self.neuralNetworkWidget)
        self.stackedWidget.addWidget(self.OptionsDataWidget)

    def switchToSelectionScreen(self):
        self.stackedWidget.setCurrentWidget(self.selectionScreen)

    def switchToBlackScholes(self):
        self.stackedWidget.setCurrentWidget(self.blackScholesWidget)

    def switchToNeuralNetwork(self):
        self.stackedWidget.setCurrentWidget(self.neuralNetworkWidget)

    def switchToOptionsWidget(self):
        self.stackedWidget.setCurrentWidget(self.OptionsDataWidget)


if __name__ == "__main__":
    app = QApplication([])
    window = MainApplication()
    window.show()
    app.exec_()

