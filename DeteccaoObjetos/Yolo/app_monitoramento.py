import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 2. Inicia o motor principal ANTES de qualquer coisa gráfica
import torch 
from ultralytics import YOLO

# 3. Importa o resto da galera
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, 
                             QFileDialog, QMessageBox, QGroupBox)
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QImage, QPixmap

# ======================================================================
# THREAD DO YOLO (Roda em segundo plano para não travar a Interface)
# ======================================================================
class YoloThread(QThread):
    # Sinal que envia o frame processado para a interface gráfica
    frame_atualizado = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._rodando = False
        self.fonte_video = 0
        self.nome_modelo = 'yolo26n.pt' # Padrão: Nano

    def run(self):
        self._rodando = True
        
        try:
            print(f"Carregando modelo {self.nome_modelo}...")
            modelo = YOLO(self.nome_modelo)
            captura = cv2.VideoCapture(self.fonte_video)
            
            if not captura.isOpened():
                print("Erro ao abrir a fonte de vídeo.")
                self._rodando = False
                return

            while self._rodando:
                sucesso, frame = captura.read()
                if not sucesso:
                    break
                
                # =====================================================
                # OTIMIZAÇÃO: Redimensionamento Proporcional (Sem deformar)
                # =====================================================
                altura_orig, largura_orig = frame.shape[:2]
                largura_alvo = 640 # Reduz a resolução para acelerar o vídeo
                
                # Calcula a matemática para manter o Aspect Ratio (Proporção)
                fator_escala = largura_alvo / float(largura_orig)
                nova_altura = int(altura_orig * fator_escala)
                
                # Agora sim o frame é redimensionado de verdade
                frame_redimensionado = cv2.resize(frame, (largura_alvo, nova_altura))
                
                # Roda a detecção no frame menor e mais leve
                resultados = modelo(frame_redimensionado, stream=True, conf=0.25, verbose=False, imgsz=largura_alvo)
                
                for resultado in resultados:
                    caixas = resultado.boxes
                    for caixa in caixas:
                        classe_id = int(caixa.cls[0])
                        confianca = float(caixa.conf[0])
                        x1, y1, x2, y2 = map(int, caixa.xyxy[0])
                        
                        # Pessoa (Verde)
                        if classe_id == 0:
                            cor = (0, 255, 0)
                            cv2.rectangle(frame_redimensionado, (x1, y1), (x2, y2), cor, 2)
                            cv2.putText(frame_redimensionado, f"Pessoa {confianca:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
                            
                        # Mochila/Bolsa (Vermelho)
                        elif classe_id in [24, 26]:
                            cor = (0, 0, 255)
                            cv2.rectangle(frame_redimensionado, (x1, y1), (x2, y2), cor, 2)
                            cv2.putText(frame_redimensionado, f"Bolsa {confianca:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

                # Envia o frame anotado para a tela principal
                self.frame_atualizado.emit(frame_redimensionado)

            captura.release()
            
        except Exception as e:
            print(f"Erro no Thread do YOLO: {e}")

    def parar(self):
        self._rodando = False
        self.wait() # Aguarda a thread fechar com segurança


# ======================================================================
# INTERFACE GRÁFICA PRINCIPAL
# ======================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Monitoramento Escolar AI")
        self.setGeometry(100, 100, 1024, 768) # Tamanho inicial da janela

        # Layout Principal (Horizontal: Vídeo na esquerda, Controles na direita)
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QHBoxLayout(widget_central)

        # --- ÁREA DO VÍDEO ---
        self.label_video = QLabel("Selecione a fonte de vídeo e clique em Iniciar.")
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setStyleSheet("background-color: #000; color: #FFF; font-size: 18px;")
        # Fixando tamanho mínimo para o vídeo não encolher sumindo
        self.label_video.setMinimumSize(800, 600) 
        layout_principal.addWidget(self.label_video, stretch=3) # Stretch 3 faz ocupar mais espaço

        # --- ÁREA DE CONTROLES ---
        layout_controles = QVBoxLayout()
        layout_principal.addLayout(layout_controles, stretch=1)

        # Grupo: Modelo YOLO
        grupo_modelo = QGroupBox("Tamanho do Modelo (IA)")
        layout_modelo = QVBoxLayout()
        self.combo_modelo = QComboBox()
        self.combo_modelo.addItems(["yolo26n.pt (Nano - Muito Rápido)", 
                                    "yolo26s.pt (Small - Rápido)", 
                                    "yolo26m.pt (Medium - Equilibrado)",
                                    "yolo26l.pt (Large - Lento/Preciso)"])
        layout_modelo.addWidget(self.combo_modelo)
        grupo_modelo.setLayout(layout_modelo)
        layout_controles.addWidget(grupo_modelo)

        # Grupo: Fonte de Vídeo
        grupo_fonte = QGroupBox("Fonte de Vídeo")
        layout_fonte = QVBoxLayout()
        
        self.btn_webcam = QPushButton("Usar Webcam (0)")
        self.btn_webcam.clicked.connect(self.selecionar_webcam)
        
        self.btn_arquivo = QPushButton("Procurar Arquivo de Vídeo (MP4)")
        self.btn_arquivo.clicked.connect(self.selecionar_arquivo)
        
        self.label_fonte_atual = QLabel("Fonte: Nenhuma selecionada")
        self.label_fonte_atual.setWordWrap(True)

        layout_fonte.addWidget(self.btn_webcam)
        layout_fonte.addWidget(self.btn_arquivo)
        layout_fonte.addWidget(self.label_fonte_atual)
        grupo_fonte.setLayout(layout_fonte)
        layout_controles.addWidget(grupo_fonte)

        # Botões de Ação
        self.btn_iniciar = QPushButton("INICIAR MONITORAMENTO")
        self.btn_iniciar.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 15px;")
        self.btn_iniciar.clicked.connect(self.iniciar_processamento)
        
        self.btn_parar = QPushButton("PARAR")
        self.btn_parar.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 15px;")
        self.btn_parar.setEnabled(False)
        self.btn_parar.clicked.connect(self.parar_processamento)

        layout_controles.addStretch() # Empurra os botões para baixo
        layout_controles.addWidget(self.btn_iniciar)
        layout_controles.addWidget(self.btn_parar)

        # Variáveis internas
        self.fonte_selecionada = None
        self.thread_yolo = YoloThread()
        self.thread_yolo.frame_atualizado.connect(self.atualizar_imagem)

    def selecionar_webcam(self):
        self.fonte_selecionada = 0
        self.label_fonte_atual.setText("Fonte: Webcam USB/Nativa (0)")

    def selecionar_arquivo(self):
        arquivo, _ = QFileDialog.getOpenFileName(self, "Selecionar Vídeo", "", "Vídeos (*.mp4 *.avi *.mkv)")
        if arquivo:
            self.fonte_selecionada = arquivo
            nome_curto = arquivo.split("/")[-1]
            self.label_fonte_atual.setText(f"Fonte: {nome_curto}")

    def iniciar_processamento(self):
        if self.fonte_selecionada is None:
            QMessageBox.warning(self, "Aviso", "Selecione uma fonte de vídeo primeiro!")
            return

        texto_modelo = self.combo_modelo.currentText()
        arquivo_modelo = texto_modelo.split(" ")[0]

        self.thread_yolo.fonte_video = self.fonte_selecionada
        self.thread_yolo.nome_modelo = arquivo_modelo
        
        self.thread_yolo.start()
        
        self.btn_iniciar.setEnabled(False)
        self.btn_parar.setEnabled(True)
        self.combo_modelo.setEnabled(False)
        self.btn_webcam.setEnabled(False)
        self.btn_arquivo.setEnabled(False)

    def parar_processamento(self):
        self.thread_yolo.parar()
        self.label_video.setText("Vídeo Parado.")
        
        self.btn_iniciar.setEnabled(True)
        self.btn_parar.setEnabled(False)
        self.combo_modelo.setEnabled(True)
        self.btn_webcam.setEnabled(True)
        self.btn_arquivo.setEnabled(True)

    def atualizar_imagem(self, frame_cv):
        """ Converte a imagem BGR do OpenCV para o formato RGB do PyQt e exibe na tela """
        rgb_image = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_por_linha = ch * w
        
        q_img = QImage(rgb_image.data, w, h, bytes_por_linha, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # O PyQt cuida de exibir o vídeo no tamanho da tela preta sem esticá-lo (Qt.KeepAspectRatio)
        pixmap_escalado = pixmap.scaled(self.label_video.width(), self.label_video.height(), Qt.KeepAspectRatio)
        self.label_video.setPixmap(pixmap_escalado)

    def closeEvent(self, event):
        self.thread_yolo.parar()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    janela = MainWindow()
    janela.show()
    sys.exit(app.exec_())