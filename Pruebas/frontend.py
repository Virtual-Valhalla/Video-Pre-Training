
# Frontend para seleccionar y mostrar la captura de una pantalla
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QComboBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import mss
import numpy as np

class ScreenCaptureWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Selector y Captura de Pantalla')
        self.sct = mss.mss()
        self.monitors = self.sct.monitors[1:]  # [0] es el monitor virtual "all"

        self.combo = QComboBox()
        for i, mon in enumerate(self.monitors):
            self.combo.addItem(f"Pantalla {i+1}: {mon['width']}x{mon['height']}")
        self.combo.currentIndexChanged.connect(self.update_monitor)

        self.label = QLabel('')
        self.label.setAlignment(Qt.AlignCenter)


        # Área de datos y piloto
        self.data_label = QLabel('Esperando datos...')
        self.data_label.setAlignment(Qt.AlignCenter)
        self.pilot_label = QLabel()
        self.pilot_label.setFixedSize(30, 30)
        self.pilot_label.setAlignment(Qt.AlignCenter)
        self.update_pilot(False)

        layout = QVBoxLayout()
        layout.addWidget(self.combo)
        layout.addWidget(self.label)
        layout.addWidget(self.data_label)
        layout.addWidget(self.pilot_label)
        self.setLayout(layout)

        self.selected_monitor = 0
        self.is_racing_game = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_screen)
        self.timer.start(50)  # 20 fps aprox

        # Simulación de detección de juego de coches (alternar cada 2 seg)
        self.detect_timer = QTimer()
        self.detect_timer.timeout.connect(self.simulate_detection)
        self.detect_timer.start(2000)

    def update_pilot(self, is_racing):
        # Dibuja un círculo verde o rojo
        color = 'green' if is_racing else 'red'
        pixmap = QPixmap(30, 30)
        pixmap.fill(Qt.transparent)
        from PyQt5.QtGui import QPainter, QColor
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(5, 5, 20, 20)
        painter.end()
        self.pilot_label.setPixmap(pixmap)

    def simulate_detection(self):
        # Simula la detección alternando el estado
        self.is_racing_game = not self.is_racing_game
        self.update_pilot(self.is_racing_game)
        if self.is_racing_game:
            self.data_label.setText('Juego de coches detectado')
        else:
            self.data_label.setText('No es un juego de coches')

    def update_monitor(self, idx):
        self.selected_monitor = idx

    def capture_screen(self):
        mon = self.monitors[self.selected_monitor]
        img = np.array(self.sct.grab(mon))
        img = img[..., :3]  # BGR
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.tobytes(), w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ScreenCaptureWidget()
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec_())
