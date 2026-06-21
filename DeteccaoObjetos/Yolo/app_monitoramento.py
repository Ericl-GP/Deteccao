# -*- coding: utf-8 -*-
"""
Sistema de Monitoramento Escolar — IFTO v2
==========================================
App local para detecção de bolsas em tempo real.
Otimizado para AMD Ryzen 3 4100 (CPU-only, ONNX Runtime).

Dependências:
    pip install ultralytics onnxruntime opencv-python PyQt5

Uso:
    python app_monitoramento.py

Estrutura de pastas esperada:
    app_monitoramento.py
    modelos/
        best.onnx          ← modelo treinado (preferencial, mais rápido na CPU)
        best.pt            ← fallback PyTorch
        yolo11n.pt         ← modelo genérico leve
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'   # evita crash com OpenMP no Windows
os.environ['ORT_DISABLE_MEMORY_PATTERN'] = '1' # estabilidade ONNX no Windows

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from collections import deque
import re

import torch
import onnxruntime as ort
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QGroupBox, QSlider,
    QCheckBox, QSpinBox, QStatusBar, QFrame, QSizePolicy,
)
from PyQt5.QtCore import pyqtSignal, QThread, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

# ─────────────────────────────────────────────────────────────────────────────
MODELOS_DIR = Path(__file__).resolve().parent / 'modelos'
MODELOS_DIR.mkdir(exist_ok=True)

# IDs COCO para bolsas (usado quando modelo genérico está ativo)
COCO_BOLSA_IDS = {24, 26, 28}   # backpack, handbag, suitcase
COCO_PESSOA_ID = 0


# ═════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS DE HARDWARE / MODELO
# ═════════════════════════════════════════════════════════════════════════════

def detectar_dispositivo_torch() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def providers_onnx_disponiveis() -> list[str]:
    """Retorna providers ONNX ordenados por preferência, disponíveis no sistema."""
    todos   = set(ort.get_available_providers())
    ordem   = [
        'TensorrtExecutionProvider',
        'CUDAExecutionProvider',
        'DmlExecutionProvider',        # DirectML — Windows AMD/Intel
        'OpenVINOExecutionProvider',
        'ROCMExecutionProvider',
        'CPUExecutionProvider',
    ]
    return [p for p in ordem if p in todos]


def criar_sessao_onnx(caminho: Path) -> tuple[ort.InferenceSession, list[str]]:
    """Cria sessão ONNX Runtime com os melhores providers disponíveis."""
    providers = providers_onnx_disponiveis()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
    opts.inter_op_num_threads = 1

    sessao = ort.InferenceSession(str(caminho), sess_options=opts, providers=providers)
    return sessao, providers


def onnx_shape_fixo(caminho: Path) -> tuple[int, int] | tuple[None, None]:
    """Retorna (largura, altura) se o modelo ONNX tiver shape fixo, senão (None, None)."""
    try:
        sess, _ = criar_sessao_onnx(caminho)
        shape = sess.get_inputs()[0].shape
        if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            return int(shape[3]), int(shape[2])   # (w, h)
    except Exception:
        pass
    return None, None


def letterbox(img: np.ndarray, w: int, h: int,
              cor: tuple = (114, 114, 114)) -> tuple[np.ndarray, float, int, int]:
    """Redimensiona mantendo proporção e adiciona padding cinza (letterbox)."""
    h0, w0 = img.shape[:2]
    escala  = min(w / w0, h / h0)
    nw, nh  = int(round(w0 * escala)), int(round(h0 * escala))
    img_r   = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w   = w - nw
    pad_h   = h - nh
    top, left = pad_h // 2, pad_w // 2
    padded  = cv2.copyMakeBorder(
        img_r, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=cor
    )
    return padded, escala, left, top


def _make_divisible(x: int, divisor: int = 32) -> int:
    """Ajusta x para o múltiplo mais próximo de `divisor`, no mínimo divisor."""
    return max(divisor, int(round(x / divisor)) * divisor)


def listar_modelos() -> list[str]:
    """Lista todos os modelos .pt e .onnx na pasta modelos/."""
    modelos = []
    rotulos = {
        'best.onnx': '⚡ best.onnx — Modelo treinado (RÁPIDO)',
        'best.pt'  : '🔵 best.pt — Modelo treinado (PyTorch)',
        'last.pt'  : '🔵 last.pt — Último checkpoint',
    }
    for arq in sorted(MODELOS_DIR.iterdir(), key=lambda x: x.name.lower()):
        if arq.suffix.lower() not in ('.pt', '.onnx'):
            continue
        label = rotulos.get(arq.name, arq.name)
        if label == arq.name:
            n = arq.name.lower()
            if   '11n' in n or 'nano' in n:   label = f'🟢 {arq.name} — Nano (muito rápido)'
            elif '11s' in n or 'small' in n:  label = f'🟡 {arq.name} — Small'
            elif '11m' in n or 'medium' in n: label = f'🟠 {arq.name} — Medium'
            elif '11l' in n or 'large' in n:  label = f'🔴 {arq.name} — Large (lento)'
        modelos.append(label)
    if not modelos:
        modelos = ['Nenhum modelo encontrado — veja a pasta modelos/']
    return modelos


# ═════════════════════════════════════════════════════════════════════════════
# THREAD DE INFERÊNCIA (roda em background)
# ═════════════════════════════════════════════════════════════════════════════

class InferenciaThread(QThread):
    """
    Thread que captura frames, roda YOLO/ONNX e emite o resultado anotado.
    Comunica com a UI via sinais PyQt5.
    """
    frame_pronto   = pyqtSignal(np.ndarray)     # frame BGR anotado
    stats_atualizadas = pyqtSignal(dict)        # métricas: fps, detecções, etc.
    erro_ocorrido  = pyqtSignal(str)            # mensagem de erro

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rodando    = False
        self.fonte_video = 0           # int = webcam, str = arquivo
        self.nome_modelo = 'best.onnx'
        self.conf        = 0.35
        self.imgsz_local = 480         # resolução de inferência (troca em tempo real)
        self.gravar      = False
        self.caminho_gravacao = ''
        self._writer: cv2.VideoWriter | None = None

    # ── ciclo principal ────────────────────────────────────────────────
    def run(self):
        self._rodando = True
        modelo        = None
        sessao_onnx: ort.InferenceSession | None = None
        caminho_modelo = MODELOS_DIR / self.nome_modelo

        # ── carrega modelo ─────────────────────────────────────────────
        try:
            if not caminho_modelo.exists():
                raise FileNotFoundError(
                    f"Modelo não encontrado: {caminho_modelo}\n"
                    "Coloque o arquivo na pasta 'modelos/' e reinicie."
                )

            if caminho_modelo.suffix.lower() == '.onnx':
                sessao_onnx, providers = criar_sessao_onnx(caminho_modelo)
                modelo_nome = f"ONNX ({', '.join(p.replace('ExecutionProvider','') for p in providers)})"
                print(f"✅ ONNX carregado: {modelo_nome}")
                inp  = sessao_onnx.get_inputs()[0]
                out  = sessao_onnx.get_outputs()[0]
                print(f"   Input  shape: {inp.shape}")
                print(f"   Output shape: {out.shape}")
                # Detecta classes do modelo (via metadata se disponível)
                meta = sessao_onnx.get_modelmeta().custom_metadata_map
                nomes_classes = {}
                if 'names' in meta:
                    import ast
                    nomes_classes = ast.literal_eval(meta['names'])
            else:
                device = detectar_dispositivo_torch()
                modelo = YOLO(str(caminho_modelo), task='detect', verbose=False)
                if device != 'cpu':
                    modelo.to(device)
                print(f"✅ PyTorch carregado no dispositivo: {device}")

        except Exception as e:
            self.erro_ocorrido.emit(str(e))
            self._rodando = False
            return

        # ── abre fonte de vídeo ────────────────────────────────────────
        cap = cv2.VideoCapture(self.fonte_video)
        if not cap.isOpened():
            self.erro_ocorrido.emit(
                f"Não foi possível abrir a fonte de vídeo: {self.fonte_video}"
            )
            self._rodando = False
            return

        # define resolução da câmera se for webcam
        if isinstance(self.fonte_video, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

        # ── fila circular de FPS ───────────────────────────────────────
        tempos: deque[float] = deque(maxlen=20)

        # ── shape fixo ONNX ────────────────────────────────────────────
        onnx_w, onnx_h = onnx_shape_fixo(caminho_modelo) if sessao_onnx else (None, None)
        nome_in = sessao_onnx.get_inputs()[0].name if sessao_onnx else None

        # ── é modelo personalizado (bolsas) ou genérico COCO? ─────────
        modelo_personalizado = self.nome_modelo.lower().startswith('best') \
                            or self.nome_modelo.lower().startswith('last')

        # ── loop de captura ────────────────────────────────────────────
        while self._rodando:
            ok, frame_orig = cap.read()
            if not ok:
                # arquivo terminou → reinicia do começo
                if isinstance(self.fonte_video, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            t_ini = time.perf_counter()

            # ── prepara frame para inferência ──────────────────────────
            if onnx_w:
                # usa o tamanho fixo do ONNX
                target_w, target_h = onnx_w, onnx_h
                frame_inf, escala, pad_x, pad_y = letterbox(frame_orig, target_w, target_h)
                w_inf, h_inf = target_w, target_h
            else:
                # garante que largura/altura sejam divisíveis pelo stride (32)
                target_w = _make_divisible(self.imgsz_local, 32)
                h0, w0 = frame_orig.shape[:2]
                target_h = int(round(h0 * (target_w / w0)))
                target_h = _make_divisible(target_h, 32)
                frame_inf, escala, pad_x, pad_y = letterbox(frame_orig, target_w, target_h)
                w_inf, h_inf = target_w, target_h

            frame_desenho = frame_orig.copy()
            h_orig, w_orig = frame_orig.shape[:2]
            n_det = 0

            # ── inferência ─────────────────────────────────────────────
            try:
                if sessao_onnx:
                    blob = frame_inf[:, :, ::-1].transpose(2, 0, 1)           # BGR→RGB→CHW
                    blob = np.ascontiguousarray(blob[None], dtype=np.float32) / 255.0
                    saidas = sessao_onnx.run(None, {nome_in: blob})

                    # saída padrão YOLO ONNX: [1, 5+nc, num_preds] ou [1, num_preds, 5+nc]
                    pred = saidas[0][0]   # shape (5+nc, N) ou (N, 5+nc)
                    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
                        pred = pred.T     # transpõe para (N, 5+nc)

                    if pred.shape[0] == 0:
                        pass
                    else:
                        nc = pred.shape[1] - 4
                        # filtra por confiança máxima de classe
                        scores = pred[:, 4:]
                        confs  = scores.max(axis=1)
                        ids    = scores.argmax(axis=1)
                        mask   = confs >= self.conf
                        pred   = pred[mask]
                        confs  = confs[mask]
                        ids    = ids[mask]

                        for i in range(len(pred)):
                            cid  = int(ids[i])
                            conf = float(confs[i])
                            cx, cy, bw, bh = pred[i, :4]

                            # desnormaliza (xywh → xyxy no espaço letterbox)
                            x1_lb = int(cx - bw / 2)
                            y1_lb = int(cy - bh / 2)
                            x2_lb = int(cx + bw / 2)
                            y2_lb = int(cy + bh / 2)

                            # remove padding e reverte escala
                            x1 = int(np.clip((x1_lb - pad_x) / escala, 0, w_orig))
                            y1 = int(np.clip((y1_lb - pad_y) / escala, 0, h_orig))
                            x2 = int(np.clip((x2_lb - pad_x) / escala, 0, w_orig))
                            y2 = int(np.clip((y2_lb - pad_y) / escala, 0, h_orig))

                            nome, cor = _classificar(cid, modelo_personalizado)
                            if nome:
                                _desenhar_caixa(frame_desenho, x1, y1, x2, y2,
                                                f"{nome} {conf:.2f}", cor)
                                n_det += 1

                else:
                    # PyTorch via Ultralytics
                    resultados = modelo(frame_inf, conf=self.conf,
                                        stream=True, verbose=False)
                    for r in resultados:
                        for box in r.boxes:
                            cid  = int(box.cls[0])
                            conf = float(box.conf[0])
                            x1l, y1l, x2l, y2l = map(int, box.xyxy[0])

                            x1 = int(np.clip((x1l - pad_x) / escala, 0, w_orig))
                            y1 = int(np.clip((y1l - pad_y) / escala, 0, h_orig))
                            x2 = int(np.clip((x2l - pad_x) / escala, 0, w_orig))
                            y2 = int(np.clip((y2l - pad_y) / escala, 0, h_orig))

                            nome, cor = _classificar(cid, modelo_personalizado)
                            if nome:
                                _desenhar_caixa(frame_desenho, x1, y1, x2, y2,
                                                f"{nome} {conf:.2f}", cor)
                                n_det += 1

            except Exception as e:
                print(f"⚠️  Erro na inferência: {e}")

            # ── FPS ────────────────────────────────────────────────────
            dt = time.perf_counter() - t_ini
            tempos.append(dt)
            fps = len(tempos) / sum(tempos) if tempos else 0

            # ── overlay de métricas ────────────────────────────────────
            _overlay_metricas(frame_desenho, fps, n_det, self.conf)

            # ── gravação ───────────────────────────────────────────────
            if self.gravar:
                if self._writer is None and self.caminho_gravacao:
                    h_f, w_f = frame_desenho.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self._writer = cv2.VideoWriter(
                        self.caminho_gravacao, fourcc, 20, (w_f, h_f)
                    )
                if self._writer:
                    self._writer.write(frame_desenho)
            elif self._writer:
                self._writer.release()
                self._writer = None

            self.frame_pronto.emit(frame_desenho)
            self.stats_atualizadas.emit({'fps': fps, 'det': n_det})

        # ── limpeza ────────────────────────────────────────────────────
        cap.release()
        if self._writer:
            self._writer.release()
            self._writer = None

    def parar(self):
        self._rodando = False
        self.wait()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de desenho (fora da classe para reuso)
# ─────────────────────────────────────────────────────────────────────────────

def _classificar(cid: int, modelo_personalizado: bool) -> tuple[str, tuple]:
    """Retorna (nome_da_classe, cor_BGR) ou ('', None) se não for relevante."""
    if modelo_personalizado:
        if cid == 0:
            return 'Bolsa', (0, 0, 220)   # vermelho
        return '', None
    else:
        if cid == COCO_PESSOA_ID:
            return 'Pessoa', (30, 200, 30)  # verde
        if cid in COCO_BOLSA_IDS:
            return 'Bolsa', (0, 0, 220)     # vermelho
        return '', None


def _desenhar_caixa(frame, x1, y1, x2, y2, texto, cor):
    """Desenha caixa com fundo semitransparente e texto legível."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), cor, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), cor, -1)
    cv2.putText(frame, texto, (x1 + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def _overlay_metricas(frame, fps: float, n_det: int, conf: float):
    """Desenha painel de métricas no canto superior esquerdo."""
    linhas = [
        f"FPS: {fps:.1f}",
        f"Bolsas: {n_det}",
        f"Conf: {conf:.2f}",
    ]
    y0, dy, pad = 22, 20, 6
    larg_max = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for l in linhas)
    cv2.rectangle(frame, (4, 4), (larg_max + pad * 2, y0 + dy * len(linhas)),
                  (20, 20, 20), -1)
    for i, linha in enumerate(linhas):
        cor = (0, 255, 120) if i == 0 else (200, 200, 200)
        cv2.putText(frame, linha, (pad, y0 + dy * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 1, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════════════════════
# INTERFACE GRÁFICA
# ═════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎒 Monitoramento Escolar — IFTO v2")
        self.setMinimumSize(1100, 680)

        self._fonte_selecionada = None
        self._gravando = False
        self._thread: InferenciaThread | None = None

        self._construir_ui()
        self._aplicar_tema()

        # Timer de status (atualiza barra inferior a cada 500 ms)
        self._timer_status = QTimer(self)
        self._timer_status.timeout.connect(self._atualizar_status_bar)
        self._fps_atual  = 0.0
        self._det_atual  = 0

    # ── construção da UI ───────────────────────────────────────────────
    def _construir_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── painel de vídeo ────────────────────────────────────────────
        self.lbl_video = QLabel("Selecione uma fonte de vídeo\ne clique em INICIAR.")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(800, 600)
        self.lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.lbl_video, stretch=4)

        # ── painel lateral de controles ────────────────────────────────
        painel = QVBoxLayout()
        painel.setSpacing(10)
        root.addLayout(painel, stretch=1)

        # Modelo
        grp_modelo = QGroupBox("Modelo de IA")
        ly_m = QVBoxLayout(grp_modelo)
        self.combo_modelo = QComboBox()
        self.combo_modelo.addItems(listar_modelos())
        ly_m.addWidget(self.combo_modelo)
        painel.addWidget(grp_modelo)

        # Fonte de vídeo
        grp_fonte = QGroupBox("Fonte de Vídeo")
        ly_f = QVBoxLayout(grp_fonte)
        self.btn_webcam  = QPushButton("📷  Webcam (0)")
        self.btn_cam1    = QPushButton("📷  Câmera (1)")
        self.btn_arquivo = QPushButton("🎬  Abrir arquivo de vídeo…")
        self.lbl_fonte   = QLabel("Nenhuma selecionada")
        self.lbl_fonte.setWordWrap(True)
        for btn in (self.btn_webcam, self.btn_cam1, self.btn_arquivo):
            ly_f.addWidget(btn)
        ly_f.addWidget(self.lbl_fonte)
        self.btn_webcam.clicked.connect(lambda: self._sel_cam(0))
        self.btn_cam1.clicked.connect(lambda: self._sel_cam(1))
        self.btn_arquivo.clicked.connect(self._sel_arquivo)
        painel.addWidget(grp_fonte)

        # Configurações
        grp_cfg = QGroupBox("Configurações")
        ly_c = QVBoxLayout(grp_cfg)

        # Confiança
        ly_c.addWidget(QLabel("Confiança mínima:"))
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 90)
        self.slider_conf.setValue(35)
        self.lbl_conf = QLabel("0.35")
        self.slider_conf.valueChanged.connect(
            lambda v: self.lbl_conf.setText(f"{v/100:.2f}")
        )
        row_conf = QHBoxLayout()
        row_conf.addWidget(self.slider_conf)
        row_conf.addWidget(self.lbl_conf)
        ly_c.addLayout(row_conf)

        # Resolução
        ly_c.addWidget(QLabel("Resolução de inferência (px largura):"))
        self.spin_res = QSpinBox()
        self.spin_res.setRange(160, 1280)
        self.spin_res.setSingleStep(32)
        self.spin_res.setValue(480)
        self.spin_res.setToolTip(
            "Reduz a imagem antes de rodar a IA.\n"
            "Menor = mais rápido (recomendado: 320–480 para CPU)."
        )
        ly_c.addWidget(self.spin_res)

        # Gravar
        self.chk_gravar = QCheckBox("Gravar vídeo anotado")
        ly_c.addWidget(self.chk_gravar)

        painel.addWidget(grp_cfg)

        # Botões de ação
        painel.addStretch()
        self.btn_iniciar = QPushButton("▶  INICIAR")
        self.btn_iniciar.setFixedHeight(48)
        self.btn_parar   = QPushButton("⏹  PARAR")
        self.btn_parar.setFixedHeight(48)
        self.btn_parar.setEnabled(False)
        self.btn_iniciar.clicked.connect(self._iniciar)
        self.btn_parar.clicked.connect(self._parar)
        painel.addWidget(self.btn_iniciar)
        painel.addWidget(self.btn_parar)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Pronto.")

    def _aplicar_tema(self):
        self.setStyleSheet("""
            QMainWindow { background: #1e1e2e; }
            QWidget     { background: #1e1e2e; color: #cdd6f4; font-size: 13px; }
            QGroupBox   { border: 1px solid #45475a; border-radius: 6px;
                          margin-top: 10px; padding-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px;
                               color: #89b4fa; font-weight: bold; }
            QPushButton { background: #313244; border: 1px solid #45475a;
                          border-radius: 5px; padding: 6px 10px; color: #cdd6f4; }
            QPushButton:hover   { background: #45475a; }
            QPushButton:pressed { background: #585b70; }
            QPushButton:disabled { color: #6c7086; }
            QComboBox, QSpinBox { background: #313244; border: 1px solid #45475a;
                                  border-radius: 4px; padding: 4px 8px; }
            QSlider::groove:horizontal { background: #45475a; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #89b4fa; width: 14px; height: 14px;
                                         margin: -5px 0; border-radius: 7px; }
            QLabel#lbl_video { background: #11111b; color: #585b70;
                               font-size: 16px; border-radius: 6px; }
            QStatusBar { background: #181825; color: #6c7086; font-size: 11px; }
        """)
        self.lbl_video.setObjectName("lbl_video")
        self.btn_iniciar.setStyleSheet(
            "QPushButton { background: #40a02b; color: white; font-weight: bold; }"
            "QPushButton:hover { background: #4ec031; }"
            "QPushButton:disabled { background: #313244; color: #6c7086; }"
        )
        self.btn_parar.setStyleSheet(
            "QPushButton { background: #d20f39; color: white; font-weight: bold; }"
            "QPushButton:hover { background: #e31b4a; }"
            "QPushButton:disabled { background: #313244; color: #6c7086; }"
        )

    # ── seleção de fonte ───────────────────────────────────────────────
    def _sel_cam(self, idx: int):
        self._fonte_selecionada = idx
        self.lbl_fonte.setText(f"Webcam ({idx})")

    def _sel_arquivo(self):
        arq, _ = QFileDialog.getOpenFileName(
            self, "Selecionar vídeo", "",
            "Vídeos (*.mp4 *.avi *.mkv *.mov *.ts *.m4v)"
        )
        if arq:
            self._fonte_selecionada = arq
            self.lbl_fonte.setText(Path(arq).name)

    # ── controle de thread ─────────────────────────────────────────────
    def _iniciar(self):
        if self._fonte_selecionada is None:
            QMessageBox.warning(self, "Aviso", "Selecione uma fonte de vídeo primeiro.")
            return

        # extrai nome do arquivo a partir do texto do combo (procura *.pt ou *.onnx)
        texto = self.combo_modelo.currentText()
        m = re.search(r"([\w\-. ]+\.(?:pt|onnx))", texto, flags=re.IGNORECASE)
        if m:
            nome_modelo = m.group(1).strip()
        else:
            # fallback: pega último token possivelmente com extensão
            tokens = texto.split()
            nome_modelo = tokens[-1] if tokens else texto

        # configura gravação se solicitado
        caminho_grav = ''
        if self.chk_gravar.isChecked():
            caminho_grav, _ = QFileDialog.getSaveFileName(
                self, "Salvar vídeo gravado", "gravacao.mp4",
                "Vídeo MP4 (*.mp4)"
            )

        self._thread = InferenciaThread()
        self._thread.fonte_video      = self._fonte_selecionada
        self._thread.nome_modelo      = nome_modelo
        self._thread.conf             = self.slider_conf.value() / 100
        self._thread.imgsz_local      = self.spin_res.value()
        self._thread.gravar           = bool(caminho_grav)
        self._thread.caminho_gravacao = caminho_grav

        self._thread.frame_pronto.connect(self._exibir_frame)
        self._thread.stats_atualizadas.connect(self._on_stats)
        self._thread.erro_ocorrido.connect(self._on_erro)
        self._thread.finished.connect(self._on_thread_finalizada)

        self._thread.start()
        self._timer_status.start(500)

        self._set_ui_rodando(True)
        self.status_bar.showMessage(f"Rodando: {nome_modelo}")

    def _parar(self):
        if self._thread and self._thread.isRunning():
            self._thread.parar()

    def _set_ui_rodando(self, rodando: bool):
        self.btn_iniciar.setEnabled(not rodando)
        self.btn_parar.setEnabled(rodando)
        for w in (self.combo_modelo, self.btn_webcam, self.btn_cam1,
                  self.btn_arquivo, self.spin_res, self.chk_gravar):
            w.setEnabled(not rodando)
        self.slider_conf.setEnabled(True)   # conf pode ser ajustada ao vivo

    # ── slots ──────────────────────────────────────────────────────────
    def _exibir_frame(self, frame_bgr: np.ndarray):
        """Atualiza o painel de vídeo com o frame anotado."""
        # Atualiza confiança ao vivo
        if self._thread:
            self._thread.conf = self.slider_conf.value() / 100

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_video.setPixmap(pix)

    def _on_stats(self, stats: dict):
        self._fps_atual  = stats.get('fps', 0)
        self._det_atual  = stats.get('det', 0)

    def _atualizar_status_bar(self):
        self.status_bar.showMessage(
            f"FPS: {self._fps_atual:.1f}   |   Bolsas detectadas: {self._det_atual}   |   "
            f"Conf: {self.slider_conf.value()/100:.2f}   |   Res: {self.spin_res.value()}px"
        )

    def _on_erro(self, msg: str):
        QMessageBox.critical(self, "Erro", msg)
        self._on_thread_finalizada()

    def _on_thread_finalizada(self):
        self._timer_status.stop()
        self._set_ui_rodando(False)
        self.lbl_video.setText("Vídeo parado.")
        self.status_bar.showMessage("Pronto.")

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            self._thread.parar()
        event.accept()


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    janela = MainWindow()
    janela.show()
    sys.exit(app.exec_())
