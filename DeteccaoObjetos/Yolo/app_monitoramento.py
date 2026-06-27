# -*- coding: utf-8 -*-
"""
Sistema de Monitoramento Escolar — IFTO v3
==========================================
App local para detecção de bolsas em tempo real.
Otimizado para AMD Ryzen 3 4100 (CPU-only, ONNX Runtime).

Dependências:
    pip install ultralytics onnxruntime opencv-python PyQt5 pyyaml

Estrutura de pastas:
    app_monitoramento.py
    modelos/            ← modelos .pt e .onnx treinados no Colab
    data/               ← YAMLs exportados do Colab (data.yaml, data_v3.yaml …)
                          O app faz o pareamento modelo ↔ YAML automaticamente.

Lógica de pareamento modelo ↔ YAML (em ordem de prioridade):
    1. data/<stem_do_modelo>.yaml   (ex: best_dual.onnx → data/best_dual.yaml)
    2. data/data_v3.yaml            (se modelo tiver 'dual'/'v3' no nome)
    3. data/data.yaml               (fallback genérico)
    4. qualquer *.yaml em data/     (primeiro encontrado)
    5. sem YAML → modo COCO genérico (backpack/handbag/suitcase + person)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']      = 'True'   # evita crash OpenMP no Windows
os.environ['ORT_DISABLE_MEMORY_PATTERN'] = '1'      # estabilidade ONNX no Windows

import sys
import time
import re
import ast
import cv2
import numpy as np
import yaml
from pathlib import Path
from collections import deque

import torch
import onnxruntime as ort
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
    QFileDialog, QMessageBox, QGroupBox, QSlider,
    QCheckBox, QSpinBox, QStatusBar, QSizePolicy,
)
from PyQt5.QtCore import pyqtSignal, QThread, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
MODELOS_DIR = BASE_DIR / 'modelos'
DATA_DIR    = BASE_DIR / 'data'
MODELOS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Fallback COCO quando nenhum YAML é encontrado
_COCO_PESSOA_IDS = {0}
_COCO_BOLSA_IDS  = {24, 26, 28}   # backpack, handbag, suitcase

# Palavras-chave para inferir o papel de cada classe pelo nome
_PALAVRAS_PESSOA = {'pessoa', 'person', 'people', 'humano', 'human'}
_PALAVRAS_BOLSA  = {
    'bolsa', 'bag', 'mochila', 'backpack',
    'handbag', 'suitcase', 'maleta', 'mala', 'sacola',
}

# Parâmetros padrão do filtro hierárquico (ajustáveis na Célula 6 do notebook)
FILTRO_LIMIAR_OVERLAP = 0.30   # IoU mínimo para ativar penalidade
FILTRO_PENALIDADE     = 0.50   # redução de confiança da bolsa sobreposta


# ═════════════════════════════════════════════════════════════════════════════
# GERENCIAMENTO DE YAML E CONFIGURAÇÃO DE CLASSES
# ═════════════════════════════════════════════════════════════════════════════

def _normalizar_names(names_raw) -> dict[int, str]:
    """Converte 'names' do YAML para {int: str} independente do formato."""
    if isinstance(names_raw, list):
        return {i: str(v) for i, v in enumerate(names_raw)}
    if isinstance(names_raw, dict):
        return {int(k): str(v) for k, v in names_raw.items()}
    return {}


def _inferir_papeis(names: dict[int, str]) -> tuple[set[int], set[int]]:
    """
    Infere quais IDs são 'pessoa' e quais são 'bolsa' a partir dos nomes.
    Usa matching por palavras-chave para ser agnóstico ao idioma.
    """
    ids_pessoa, ids_bolsa = set(), set()
    for cid, nome in names.items():
        n = nome.lower().strip()
        if any(p in n for p in _PALAVRAS_PESSOA):
            ids_pessoa.add(cid)
        elif any(p in n for p in _PALAVRAS_BOLSA):
            ids_bolsa.add(cid)
    return ids_pessoa, ids_bolsa


def carregar_yaml_modelo(nome_modelo: str) -> dict | None:
    """
    Procura o YAML correspondente ao modelo na pasta data/.
    Retorna o dict do YAML ou None se não encontrar.

    Prioridade:
      1. data/<stem>.yaml          (best_dual.onnx → data/best_dual.yaml)
      2. data/data_v3.yaml         (se 'dual' ou 'v3' no nome do modelo)
      3. data/data.yaml
      4. qualquer *.yaml em data/
    """
    stem = Path(nome_modelo).stem   # 'best_dual' de 'best_dual.onnx'
    candidatos = [DATA_DIR / f"{stem}.yaml"]

    nome_lower = nome_modelo.lower()
    if 'dual' in nome_lower or 'v3' in nome_lower:
        candidatos.append(DATA_DIR / 'data_v3.yaml')

    candidatos.append(DATA_DIR / 'data.yaml')

    # qualquer yaml na pasta
    for arq in sorted(DATA_DIR.glob('*.yaml')):
        if arq not in candidatos:
            candidatos.append(arq)

    for c in candidatos:
        if c.exists():
            try:
                with open(c, encoding='utf-8') as f:
                    dados = yaml.safe_load(f)
                if isinstance(dados, dict) and 'names' in dados:
                    dados['_yaml_path'] = c
                    return dados
            except Exception as e:
                print(f"⚠️  Erro ao ler YAML {c.name}: {e}")

    return None


def construir_config_classes(nome_modelo: str) -> dict:
    """
    Monta o dicionário de configuração de classes para um modelo.

    Retorna:
        names            : {int: str}  — mapa id → nome da classe
        ids_pessoa       : set[int]    — IDs classificados como 'pessoa'
        ids_bolsa        : set[int]    — IDs classificados como 'bolsa'
        usar_filtro      : bool        — True se ambas as classes existem
        nc               : int
        yaml_path        : Path | None
        yaml_nome        : str
        limiar_overlap   : float
        penalidade       : float
    """
    yaml_dados = carregar_yaml_modelo(nome_modelo)

    if yaml_dados:
        names      = _normalizar_names(yaml_dados.get('names', {}))
        nc         = int(yaml_dados.get('nc', len(names)))
        yaml_path  = yaml_dados.get('_yaml_path')
        yaml_nome  = yaml_path.name if yaml_path else '?'
        ids_pessoa, ids_bolsa = _inferir_papeis(names)

        # Se o YAML não tem palavras-chave reconhecíveis, trata tudo como bolsa
        if not ids_bolsa and not ids_pessoa:
            ids_bolsa = set(names.keys())

        usar_filtro = bool(ids_pessoa and ids_bolsa)

        return {
            'names'         : names,
            'ids_pessoa'    : ids_pessoa,
            'ids_bolsa'     : ids_bolsa,
            'usar_filtro'   : usar_filtro,
            'nc'            : nc,
            'yaml_path'     : yaml_path,
            'yaml_nome'     : yaml_nome,
            'limiar_overlap': FILTRO_LIMIAR_OVERLAP,
            'penalidade'    : FILTRO_PENALIDADE,
            'modo'          : 'personalizado',
        }

    # ── Fallback: COCO genérico ───────────────────────────────────────────────
    names_coco = {
        0 : 'Pessoa',
        24: 'Bolsa',
        26: 'Bolsa',
        28: 'Bolsa',
    }
    return {
        'names'         : names_coco,
        'ids_pessoa'    : _COCO_PESSOA_IDS,
        'ids_bolsa'     : _COCO_BOLSA_IDS,
        'usar_filtro'   : True,
        'nc'            : 80,
        'yaml_path'     : None,
        'yaml_nome'     : 'COCO (sem YAML)',
        'limiar_overlap': FILTRO_LIMIAR_OVERLAP,
        'penalidade'    : FILTRO_PENALIDADE,
        'modo'          : 'coco',
    }


def resumo_config(config: dict) -> str:
    """Gera texto curto descrevendo a configuração de classes carregada."""
    nc     = config['nc']
    yaml_n = config['yaml_nome']
    nomes  = list(config['names'].values())
    nomes_unicos = list(dict.fromkeys(nomes))   # preserva ordem, remove duplicatas
    filtro = '✅ filtro ativo' if config['usar_filtro'] else '❌ filtro inativo'
    return f"📄 {yaml_n}  ·  {nc} classe(s): {', '.join(nomes_unicos)}  ·  {filtro}"


# ═════════════════════════════════════════════════════════════════════════════
# FILTRO HIERÁRQUICO (idêntico ao notebook v3, Célula 6)
# ═════════════════════════════════════════════════════════════════════════════

def _iou_boxes(b1: list, b2: list) -> float:
    """IoU entre dois bounding boxes [x1, y1, x2, y2]."""
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def filtrar_hierarquico(
    dets_brutas : list[dict],
    ids_pessoa  : set[int],
    ids_bolsa   : set[int],
    conf_bolsa  : float,
    conf_pessoa : float,
    limiar      : float = FILTRO_LIMIAR_OVERLAP,
    penalidade  : float = FILTRO_PENALIDADE,
) -> list[dict]:
    """
    Aplica filtro hierárquico pessoa → bolsa.

    Lógica:
        Para cada bolsa detectada:
            iou_max = maior IoU com qualquer caixa de pessoa
            se iou_max >= limiar:
                conf_efetiva = conf_bolsa × (1 − penalidade)
                se conf_efetiva < conf_bolsa: descarta
            senão: aceita normalmente

    Retorna lista de dets com campo extra 'penalizada': bool.
    """
    pessoas = [d for d in dets_brutas
               if d['cid'] in ids_pessoa and d['conf'] >= conf_pessoa]
    bolsas  = [d for d in dets_brutas
               if d['cid'] in ids_bolsa  and d['conf'] >= conf_bolsa]

    finais = list(pessoas)

    for b in bolsas:
        iou_max = 0.0
        for p in pessoas:
            iou_max = max(iou_max, _iou_boxes(b['xyxy'], p['xyxy']))

        conf_ef = b['conf']
        penalizada = False

        if iou_max >= limiar:
            conf_ef    = b['conf'] * (1.0 - penalidade)
            penalizada = True

        if conf_ef >= conf_bolsa:
            finais.append({**b, 'conf': conf_ef, 'penalizada': penalizada})

    return finais


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
    todos = set(ort.get_available_providers())
    ordem = [
        'TensorrtExecutionProvider',
        'CUDAExecutionProvider',
        'DmlExecutionProvider',
        'OpenVINOExecutionProvider',
        'ROCMExecutionProvider',
        'CPUExecutionProvider',
    ]
    return [p for p in ordem if p in todos]


def criar_sessao_onnx(caminho: Path) -> tuple[ort.InferenceSession, list[str]]:
    providers = providers_onnx_disponiveis()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
    opts.inter_op_num_threads = 1
    sessao = ort.InferenceSession(str(caminho), sess_options=opts, providers=providers)
    return sessao, providers


def onnx_shape_fixo(caminho: Path) -> tuple[int, int] | tuple[None, None]:
    try:
        sess, _ = criar_sessao_onnx(caminho)
        shape = sess.get_inputs()[0].shape
        if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            return int(shape[3]), int(shape[2])
    except Exception:
        pass
    return None, None


def letterbox(
    img: np.ndarray, w: int, h: int,
    cor: tuple = (114, 114, 114)
) -> tuple[np.ndarray, float, int, int]:
    h0, w0  = img.shape[:2]
    escala  = min(w / w0, h / h0)
    nw, nh  = int(round(w0 * escala)), int(round(h0 * escala))
    img_r   = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = w - nw, h - nh
    top,  left   = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(
        img_r, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=cor,
    )
    return padded, escala, left, top


def _make_divisible(x: int, div: int = 32) -> int:
    return max(div, int(round(x / div)) * div)


def listar_modelos() -> list[str]:
    """Lista modelos da pasta modelos/ com indicador de YAML disponível."""
    if not any(MODELOS_DIR.iterdir().__next__
               for _ in [None]):      # pasta vazia
        pass

    ROTULOS = {
        'best_dual.onnx': '⚡ best_dual.onnx — Dual-classe (RÁPIDO)',
        'best_dual.pt'  : '🔵 best_dual.pt  — Dual-classe (PyTorch)',
        'best.onnx'     : '⚡ best.onnx      — Modelo treinado (RÁPIDO)',
        'best.pt'       : '🔵 best.pt        — Modelo treinado (PyTorch)',
        'last.pt'       : '🔵 last.pt        — Último checkpoint',
    }

    modelos = []
    try:
        for arq in sorted(MODELOS_DIR.iterdir(), key=lambda x: x.name.lower()):
            if arq.suffix.lower() not in ('.pt', '.onnx'):
                continue

            label = ROTULOS.get(arq.name)
            if not label:
                n = arq.name.lower()
                if   'nano'  in n or '26n' in n or '11n' in n: icone = '🟢'
                elif 'small' in n or '26s' in n or '11s' in n: icone = '🟡'
                elif 'med'   in n or '26m' in n or '11m' in n: icone = '🟠'
                elif 'large' in n or '26l' in n or '11l' in n: icone = '🔴'
                else:                                            icone = '🔵'
                label = f"{icone} {arq.name}"

            # Indica se há YAML associado
            yaml_d = carregar_yaml_modelo(arq.name)
            if yaml_d:
                nc    = yaml_d.get('nc', '?')
                yname = yaml_d['_yaml_path'].name
                label += f"  [{nc}cls · {yname}]"

            modelos.append(label)
    except StopIteration:
        pass

    if not modelos:
        modelos = ['Nenhum modelo em modelos/ — faça download do Colab']
    return modelos


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS DE DESENHO
# ═════════════════════════════════════════════════════════════════════════════

# Paleta fixa por papel da classe
_COR_BOLSA    = (0,   0,   220)   # vermelho
_COR_PESSOA   = (30,  200,  30)   # verde
_COR_PENALIZ  = (0,   165, 255)   # laranja (bolsa penalizada, ainda aceita)
_COR_GENERICO = (200, 200, 200)   # cinza (classe sem papel definido)


def _cor_deteccao(cid: int, penalizada: bool, config: dict) -> tuple:
    if penalizada:
        return _COR_PENALIZ
    if cid in config['ids_pessoa']:
        return _COR_PESSOA
    if cid in config['ids_bolsa']:
        return _COR_BOLSA
    return _COR_GENERICO


def _desenhar_caixa(frame, x1, y1, x2, y2, texto, cor):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), cor, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = max(y1 - 4, th + 4)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), cor, -1)
    cv2.putText(frame, texto, (x1 + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def _overlay_metricas(frame, fps: float, n_bolsas: int, conf: float,
                      n_pessoas: int = 0, filtro_ativo: bool = False):
    linhas = [f"FPS: {fps:.1f}"]
    if filtro_ativo:
        linhas += [f"Bolsas: {n_bolsas}", f"Pessoas: {n_pessoas}"]
    else:
        linhas += [f"Bolsas: {n_bolsas}"]
    linhas.append(f"Conf: {conf:.2f}")

    y0, dy, pad = 22, 20, 6
    larg = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for l in linhas)
    cv2.rectangle(frame, (4, 4), (larg + pad * 2, y0 + dy * len(linhas)), (20, 20, 20), -1)
    for i, linha in enumerate(linhas):
        cor_txt = (0, 255, 120) if i == 0 else (200, 200, 200)
        cv2.putText(frame, linha, (pad, y0 + dy * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_txt, 1, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════════════════════
# THREAD DE INFERÊNCIA
# ═════════════════════════════════════════════════════════════════════════════

class InferenciaThread(QThread):
    frame_pronto      = pyqtSignal(np.ndarray)
    stats_atualizadas = pyqtSignal(dict)
    erro_ocorrido     = pyqtSignal(str)
    config_carregada  = pyqtSignal(dict)   # emite config de classes ao iniciar

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rodando         = False
        self.fonte_video      = 0
        self.nome_modelo      = 'best_dual.onnx'
        self.conf             = 0.30
        self.imgsz_local      = 480
        self.gravar           = False
        self.caminho_gravacao = ''
        self._writer: cv2.VideoWriter | None = None

    def run(self):
        self._rodando  = True
        modelo         = None
        sessao_onnx: ort.InferenceSession | None = None
        caminho_modelo = MODELOS_DIR / self.nome_modelo

        # ── 1. Carrega configuração de classes via YAML ────────────────
        config = construir_config_classes(self.nome_modelo)
        self.config_carregada.emit(config)
        print(f"\n📋 Config de classes: {resumo_config(config)}")
        print(f"   ids_pessoa={config['ids_pessoa']}  ids_bolsa={config['ids_bolsa']}")

        # ── 2. Carrega modelo ──────────────────────────────────────────
        try:
            if not caminho_modelo.exists():
                raise FileNotFoundError(
                    f"Modelo não encontrado: {caminho_modelo}\n"
                    "Baixe o modelo do Colab e coloque em modelos/"
                )

            if caminho_modelo.suffix.lower() == '.onnx':
                sessao_onnx, providers = criar_sessao_onnx(caminho_modelo)
                prov_str = ', '.join(p.replace('ExecutionProvider', '') for p in providers)
                print(f"✅ ONNX carregado [{prov_str}]")
                inp = sessao_onnx.get_inputs()[0]
                out = sessao_onnx.get_outputs()[0]
                print(f"   Input : {inp.shape}  Output: {out.shape}")

                # Tenta enriquecer config com metadata embutida no ONNX
                meta = sessao_onnx.get_modelmeta().custom_metadata_map
                if 'names' in meta and config['modo'] == 'coco':
                    try:
                        names_meta = ast.literal_eval(meta['names'])
                        names_norm = _normalizar_names(names_meta)
                        ids_p, ids_b = _inferir_papeis(names_norm)
                        if ids_b:   # só usa se reconheceu bolsas
                            config['names']       = names_norm
                            config['ids_pessoa']  = ids_p
                            config['ids_bolsa']   = ids_b
                            config['usar_filtro'] = bool(ids_p and ids_b)
                            config['nc']          = len(names_norm)
                            config['yaml_nome']  += ' + metadata ONNX'
                            print(f"   Metadata ONNX usada: {names_norm}")
                    except Exception:
                        pass
            else:
                device = detectar_dispositivo_torch()
                modelo = YOLO(str(caminho_modelo), task='detect', verbose=False)
                if device != 'cpu':
                    modelo.to(device)
                print(f"✅ PyTorch carregado [{device}]")

        except Exception as e:
            self.erro_ocorrido.emit(str(e))
            self._rodando = False
            return

        # ── 3. Abre fonte de vídeo ────────────────────────────────────
        cap = cv2.VideoCapture(self.fonte_video)
        if not cap.isOpened():
            self.erro_ocorrido.emit(
                f"Não foi possível abrir: {self.fonte_video}"
            )
            self._rodando = False
            return

        if isinstance(self.fonte_video, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
            cap.set(cv2.CAP_PROP_FPS, 30)

        tempos: deque[float] = deque(maxlen=20)
        onnx_w, onnx_h = onnx_shape_fixo(caminho_modelo) if sessao_onnx else (None, None)
        nome_in = sessao_onnx.get_inputs()[0].name if sessao_onnx else None

        # Confiança mínima para capturar pessoas (ligeiramente abaixo da bolsa)
        # usada como limiar inicial de coleta no loop
        def conf_pessoa_thr():
            return max(0.10, self.conf - 0.05)

        # ── 4. Loop de captura ────────────────────────────────────────
        while self._rodando:
            ok, frame_orig = cap.read()
            if not ok:
                if isinstance(self.fonte_video, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            t_ini = time.perf_counter()

            # Prepara frame para inferência (letterbox)
            if onnx_w:
                tw, th = onnx_w, onnx_h
            else:
                tw = _make_divisible(self.imgsz_local, 32)
                h0, w0 = frame_orig.shape[:2]
                th = _make_divisible(int(h0 * tw / w0), 32)

            frame_inf, escala, pad_x, pad_y = letterbox(frame_orig, tw, th)
            frame_desenho = frame_orig.copy()
            h_orig, w_orig = frame_orig.shape[:2]

            # Limiar de coleta: pega ambos pessoa e bolsa desde já
            conf_min_coleta = min(self.conf, conf_pessoa_thr())

            # ── 4a. Inferência → coleta dets brutas ───────────────────
            dets_brutas: list[dict] = []

            try:
                if sessao_onnx:
                    blob = frame_inf[:, :, ::-1].transpose(2, 0, 1)
                    blob = np.ascontiguousarray(blob[None], dtype=np.float32) / 255.0
                    saidas = sessao_onnx.run(None, {nome_in: blob})

                    pred = saidas[0][0]
                    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
                        pred = pred.T   # (N, 4+nc)

                    if pred.shape[0] > 0:
                        scores = pred[:, 4:]
                        confs  = scores.max(axis=1)
                        ids    = scores.argmax(axis=1)
                        mask   = confs >= conf_min_coleta

                        for i in np.where(mask)[0]:
                            cid  = int(ids[i])
                            conf = float(confs[i])
                            cx, cy, bw, bh = pred[i, :4]

                            x1_lb = int(cx - bw / 2); y1_lb = int(cy - bh / 2)
                            x2_lb = int(cx + bw / 2); y2_lb = int(cy + bh / 2)

                            x1 = int(np.clip((x1_lb - pad_x) / escala, 0, w_orig))
                            y1 = int(np.clip((y1_lb - pad_y) / escala, 0, h_orig))
                            x2 = int(np.clip((x2_lb - pad_x) / escala, 0, w_orig))
                            y2 = int(np.clip((y2_lb - pad_y) / escala, 0, h_orig))

                            # Aceita apenas classes conhecidas (ids_pessoa + ids_bolsa)
                            ids_relevantes = config['ids_pessoa'] | config['ids_bolsa']
                            if config['modo'] == 'coco' or cid in ids_relevantes:
                                dets_brutas.append({
                                    'cid' : cid, 'conf': conf,
                                    'xyxy': [x1, y1, x2, y2],
                                    'penalizada': False,
                                })

                else:
                    # PyTorch via Ultralytics
                    resultados = modelo(
                        frame_inf, conf=conf_min_coleta, stream=True, verbose=False
                    )
                    for r in resultados:
                        for box in r.boxes:
                            cid  = int(box.cls[0])
                            conf = float(box.conf[0])
                            x1l, y1l, x2l, y2l = map(int, box.xyxy[0])
                            x1 = int(np.clip((x1l - pad_x) / escala, 0, w_orig))
                            y1 = int(np.clip((y1l - pad_y) / escala, 0, h_orig))
                            x2 = int(np.clip((x2l - pad_x) / escala, 0, w_orig))
                            y2 = int(np.clip((y2l - pad_y) / escala, 0, h_orig))

                            ids_relevantes = config['ids_pessoa'] | config['ids_bolsa']
                            if config['modo'] == 'coco' or cid in ids_relevantes:
                                dets_brutas.append({
                                    'cid' : cid, 'conf': conf,
                                    'xyxy': [x1, y1, x2, y2],
                                    'penalizada': False,
                                })

            except Exception as e:
                print(f"⚠️  Erro na inferência: {e}")

            # ── 4b. Aplica filtro hierárquico (se configurado) ─────────
            if config['usar_filtro']:
                dets_finais = filtrar_hierarquico(
                    dets_brutas,
                    ids_pessoa  = config['ids_pessoa'],
                    ids_bolsa   = config['ids_bolsa'],
                    conf_bolsa  = self.conf,
                    conf_pessoa = conf_pessoa_thr(),
                    limiar      = config['limiar_overlap'],
                    penalidade  = config['penalidade'],
                )
            else:
                # Sem filtro: mostra apenas bolsas com conf >= threshold
                dets_finais = [
                    d for d in dets_brutas
                    if d['cid'] in config['ids_bolsa'] and d['conf'] >= self.conf
                ]

            # ── 4c. Desenha detecções ──────────────────────────────────
            n_bolsas  = 0
            n_pessoas = 0

            for d in dets_finais:
                cid  = d['cid']
                conf = d['conf']
                nome = config['names'].get(cid, str(cid))
                cor  = _cor_deteccao(cid, d.get('penalizada', False), config)
                x1, y1, x2, y2 = d['xyxy']

                sufixo = ' ⚠' if d.get('penalizada') else ''
                _desenhar_caixa(frame_desenho, x1, y1, x2, y2,
                                f"{nome} {conf:.2f}{sufixo}", cor)

                if cid in config['ids_bolsa']:
                    n_bolsas += 1
                elif cid in config['ids_pessoa']:
                    n_pessoas += 1

            # ── 4d. Overlay de métricas ────────────────────────────────
            dt  = time.perf_counter() - t_ini
            tempos.append(dt)
            fps = len(tempos) / sum(tempos) if tempos else 0

            _overlay_metricas(
                frame_desenho, fps, n_bolsas, self.conf,
                n_pessoas    = n_pessoas,
                filtro_ativo = config['usar_filtro'],
            )

            # ── 4e. Gravação ───────────────────────────────────────────
            if self.gravar:
                if self._writer is None and self.caminho_gravacao:
                    hf, wf = frame_desenho.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self._writer = cv2.VideoWriter(
                        self.caminho_gravacao, fourcc, 20, (wf, hf)
                    )
                if self._writer:
                    self._writer.write(frame_desenho)
            elif self._writer:
                self._writer.release()
                self._writer = None

            self.frame_pronto.emit(frame_desenho)
            self.stats_atualizadas.emit({
                'fps'     : fps,
                'bolsas'  : n_bolsas,
                'pessoas' : n_pessoas,
                'filtro'  : config['usar_filtro'],
            })

        # ── Limpeza ────────────────────────────────────────────────────
        cap.release()
        if self._writer:
            self._writer.release()
            self._writer = None

    def parar(self):
        self._rodando = False
        self.wait()


# ═════════════════════════════════════════════════════════════════════════════
# INTERFACE GRÁFICA
# ═════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎒 Monitoramento Escolar — IFTO v3")
        self.setMinimumSize(1100, 680)

        self._fonte_selecionada = None
        self._thread: InferenciaThread | None = None
        self._fps_atual    = 0.0
        self._bolsas_atual = 0
        self._pessoas_atual = 0
        self._filtro_ativo  = False

        self._construir_ui()
        self._aplicar_tema()

        self._timer_status = QTimer(self)
        self._timer_status.timeout.connect(self._atualizar_status_bar)

        # Atualiza info YAML ao iniciar
        self._on_modelo_changed()

    # ── Construção da UI ──────────────────────────────────────────────
    def _construir_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Painel de vídeo
        self.lbl_video = QLabel("Selecione uma fonte de vídeo\ne clique em INICIAR.")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(800, 560)
        self.lbl_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.lbl_video, stretch=4)

        # Painel lateral
        painel = QVBoxLayout()
        painel.setSpacing(8)
        root.addLayout(painel, stretch=1)

        # ── Modelo ────────────────────────────────────────────────────
        grp_modelo = QGroupBox("Modelo de IA")
        ly_m = QVBoxLayout(grp_modelo)
        self.combo_modelo = QComboBox()
        self.combo_modelo.addItems(listar_modelos())
        self.combo_modelo.currentIndexChanged.connect(self._on_modelo_changed)
        ly_m.addWidget(self.combo_modelo)

        # Label de informação sobre YAML + classes
        self.lbl_yaml_info = QLabel("Carregando...")
        self.lbl_yaml_info.setWordWrap(True)
        self.lbl_yaml_info.setStyleSheet(
            "font-size: 11px; color: #a6e3a1; padding: 4px;"
            "background: #1e1e2e; border-radius: 4px;"
        )
        ly_m.addWidget(self.lbl_yaml_info)

        # Legenda de cores (atualizada dinamicamente)
        self.lbl_legenda = QLabel()
        self.lbl_legenda.setWordWrap(True)
        self.lbl_legenda.setStyleSheet("font-size: 10px; color: #9399b2; padding: 2px;")
        ly_m.addWidget(self.lbl_legenda)

        painel.addWidget(grp_modelo)

        # ── Fonte de vídeo ────────────────────────────────────────────
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

        # ── Configurações ─────────────────────────────────────────────
        grp_cfg = QGroupBox("Configurações")
        ly_c = QVBoxLayout(grp_cfg)

        ly_c.addWidget(QLabel("Confiança mínima (bolsa):"))
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(10, 90)
        self.slider_conf.setValue(30)
        self.lbl_conf = QLabel("0.30")
        self.slider_conf.valueChanged.connect(
            lambda v: self.lbl_conf.setText(f"{v/100:.2f}")
        )
        row_c = QHBoxLayout()
        row_c.addWidget(self.slider_conf)
        row_c.addWidget(self.lbl_conf)
        ly_c.addLayout(row_c)

        ly_c.addWidget(QLabel("Resolução de inferência (px largura):"))
        self.spin_res = QSpinBox()
        self.spin_res.setRange(160, 1280)
        self.spin_res.setSingleStep(32)
        self.spin_res.setValue(480)
        self.spin_res.setToolTip(
            "Menor = mais rápido.\nRecomendado: 320–480 px para CPU."
        )
        ly_c.addWidget(self.spin_res)

        self.chk_gravar = QCheckBox("Gravar vídeo anotado")
        ly_c.addWidget(self.chk_gravar)

        painel.addWidget(grp_cfg)

        # ── Botões ────────────────────────────────────────────────────
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
            QPushButton:hover    { background: #45475a; }
            QPushButton:pressed  { background: #585b70; }
            QPushButton:disabled { color: #6c7086; }
            QComboBox, QSpinBox  { background: #313244; border: 1px solid #45475a;
                                   border-radius: 4px; padding: 4px 8px; }
            QSlider::groove:horizontal { background: #45475a; height: 4px;
                                         border-radius: 2px; }
            QSlider::handle:horizontal { background: #89b4fa; width: 14px;
                                         height: 14px; margin: -5px 0;
                                         border-radius: 7px; }
            QLabel#lbl_video   { background: #11111b; color: #585b70;
                                 font-size: 16px; border-radius: 6px; }
            QStatusBar { background: #181825; color: #6c7086; font-size: 11px; }
        """)
        self.lbl_video.setObjectName("lbl_video")
        self.btn_iniciar.setStyleSheet(
            "QPushButton { background: #40a02b; color: white; font-weight: bold; }"
            "QPushButton:hover    { background: #4ec031; }"
            "QPushButton:disabled { background: #313244; color: #6c7086; }"
        )
        self.btn_parar.setStyleSheet(
            "QPushButton { background: #d20f39; color: white; font-weight: bold; }"
            "QPushButton:hover    { background: #e31b4a; }"
            "QPushButton:disabled { background: #313244; color: #6c7086; }"
        )

    # ── Seleção de fonte ──────────────────────────────────────────────
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

    # ── Atualiza info YAML quando combo muda ──────────────────────────
    def _on_modelo_changed(self):
        texto = self.combo_modelo.currentText()
        m = re.search(r"([\w\-. ]+\.(?:pt|onnx))", texto, flags=re.IGNORECASE)
        nome = m.group(1).strip() if m else ''
        if not nome:
            self.lbl_yaml_info.setText("⚠️ Nenhum modelo reconhecido")
            self.lbl_legenda.setText("")
            return

        config = construir_config_classes(nome)

        # Info YAML
        nomes_unicos = list(dict.fromkeys(config['names'].values()))
        filtro_txt = "✅ filtro hierárquico ativo" if config['usar_filtro'] else "❌ filtro inativo"
        info = (
            f"📄 {config['yaml_nome']}\n"
            f"   {config['nc']} classe(s): {', '.join(nomes_unicos)}\n"
            f"   {filtro_txt}"
        )
        self.lbl_yaml_info.setText(info)

        # Legenda de cores
        leg_partes = []
        if config['ids_bolsa']:
            nome_b = config['names'].get(next(iter(config['ids_bolsa'])), 'Bolsa')
            leg_partes.append(f"🔴 {nome_b}")
        if config['ids_pessoa']:
            nome_p = config['names'].get(next(iter(config['ids_pessoa'])), 'Pessoa')
            leg_partes.append(f"🟢 {nome_p}")
        if config['usar_filtro']:
            leg_partes.append("🟠 Penalizada")
        self.lbl_legenda.setText("  ".join(leg_partes))

    # ── Controle da thread ────────────────────────────────────────────
    def _iniciar(self):
        if self._fonte_selecionada is None:
            QMessageBox.warning(self, "Aviso", "Selecione uma fonte de vídeo primeiro.")
            return

        texto = self.combo_modelo.currentText()
        m = re.search(r"([\w\-. ]+\.(?:pt|onnx))", texto, flags=re.IGNORECASE)
        nome_modelo = m.group(1).strip() if m else texto.split()[0]

        caminho_grav = ''
        if self.chk_gravar.isChecked():
            caminho_grav, _ = QFileDialog.getSaveFileName(
                self, "Salvar vídeo", "gravacao.mp4", "MP4 (*.mp4)"
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
        self._thread.config_carregada.connect(self._on_config_carregada)
        self._thread.finished.connect(self._on_thread_finalizada)

        self._thread.start()
        self._timer_status.start(500)
        self._set_ui_rodando(True)
        self.status_bar.showMessage(f"Iniciando: {nome_modelo}…")

    def _parar(self):
        if self._thread and self._thread.isRunning():
            self._thread.parar()

    def _set_ui_rodando(self, rodando: bool):
        self.btn_iniciar.setEnabled(not rodando)
        self.btn_parar.setEnabled(rodando)
        for w in (self.combo_modelo, self.btn_webcam, self.btn_cam1,
                  self.btn_arquivo, self.spin_res, self.chk_gravar):
            w.setEnabled(not rodando)
        self.slider_conf.setEnabled(True)   # ajustável ao vivo

    # ── Slots ─────────────────────────────────────────────────────────
    def _exibir_frame(self, frame_bgr: np.ndarray):
        if self._thread:
            self._thread.conf = self.slider_conf.value() / 100
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_video.setPixmap(pix)

    def _on_stats(self, stats: dict):
        self._fps_atual     = stats.get('fps',     0.0)
        self._bolsas_atual  = stats.get('bolsas',  0)
        self._pessoas_atual = stats.get('pessoas', 0)
        self._filtro_ativo  = stats.get('filtro',  False)

    def _on_config_carregada(self, config: dict):
        """Atualiza o painel de info assim que o modelo é carregado na thread."""
        nomes_unicos = list(dict.fromkeys(config['names'].values()))
        filtro_txt   = "✅ filtro ativo" if config['usar_filtro'] else "❌ filtro inativo"
        info = (
            f"📄 {config['yaml_nome']}\n"
            f"   {config['nc']} classe(s): {', '.join(nomes_unicos)}\n"
            f"   {filtro_txt}"
        )
        self.lbl_yaml_info.setText(info)

    def _atualizar_status_bar(self):
        msg = (
            f"FPS: {self._fps_atual:.1f}   |   "
            f"Bolsas: {self._bolsas_atual}"
        )
        if self._filtro_ativo:
            msg += f"   |   Pessoas: {self._pessoas_atual}   |   Filtro: ON"
        msg += (
            f"   |   Conf: {self.slider_conf.value()/100:.2f}"
            f"   |   Res: {self.spin_res.value()}px"
        )
        self.status_bar.showMessage(msg)

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