from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton

class SelectionScreen(QWidget):
    def __init__(self, switch_to_black_scholes_callback, switch_to_neural_network_callback, switch_to_options_data_callback):
        super().__init__()
        layout = QVBoxLayout()

        # Sets the background color of the layout to white
        self.setStyleSheet("background-color: white;")

        bsButton = QPushButton("Black Scholes GUI")
        bsButton.clicked.connect(switch_to_black_scholes_callback)
        # Sets the button color to orange with white text
        bsButton.setStyleSheet("QPushButton {background-color: #FFA500; color: white;}")
        layout.addWidget(bsButton)

        nnButton = QPushButton("Neural Network GUI")
        nnButton.clicked.connect(switch_to_neural_network_callback)
        nnButton.setStyleSheet("QPushButton {background-color: #FFA500; color: white;}")
        layout.addWidget(nnButton)

        osButton = QPushButton("Neural Network - Options Data")
        osButton.clicked.connect(switch_to_options_data_callback)
        osButton.setStyleSheet("QPushButton {background-color: #FFA500; color: white;}")
        layout.addWidget(osButton)

        self.setLayout(layout)
