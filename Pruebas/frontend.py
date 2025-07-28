
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


        # Carga tu modelo PyTorch aquí (ajusta la ruta y clase según tu caso)
        import torch
        from torchvision import transforms
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Cambia la ruta y la clase del modelo según tu entrenamiento
        self.modelo = torch.load('modelo_vpt_coche.pt', map_location=self.device)
        self.modelo.eval()
        # Preprocesado estándar (ajusta tamaño y normalización según tu modelo)
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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


    def es_juego_de_coches(self, img):
        # img: numpy array (H, W, 3) en BGR
        import torch
        # Convertir BGR a RGB
        img_rgb = img[..., ::-1]
        # Preprocesar
        input_tensor = self.preprocess(img_rgb)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.modelo(input_tensor)
            # Si el modelo devuelve logits, aplicar sigmoid/softmax según corresponda
            if output.shape[-1] == 1:
                prob = torch.sigmoid(output).item()
                return prob > 0.5
            else:
                prob = torch.softmax(output, dim=1)[0,1].item()
                return prob > 0.5

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

        # --- Detección IA en tiempo real ---
        es_coche = self.es_juego_de_coches(img)
        self.update_pilot(es_coche)
        if es_coche:
            self.data_label.setText('Juego de coches detectado')
        else:
            self.data_label.setText('No es un juego de coches')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ScreenCaptureWidget()
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec_())
