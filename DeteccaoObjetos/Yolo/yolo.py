import cv2
from ultralytics import YOLO

def treinar_modelo(caminho_yaml: str, epocas: int = 50, tamanho_img: int = 640):
    """
    Treina um modelo YOLOv8 do zero ou a partir de um modelo pré-treinado.
    
    Args:
        caminho_yaml (str): Caminho para o arquivo .yaml com as configurações do dataset.
        epocas (int): Número de épocas (ciclos de treinamento). Padrão é 50.
        tamanho_img (int): Tamanho das imagens para o treinamento. Padrão é 640.
    """
    print("Iniciando o carregamento do modelo base...")
    # Carrega um modelo pré-treinado pequeno (yolov8n.pt) para fazer Transfer Learning.
    # Modelos 'n' (nano) são mais rápidos e excelentes para hardware mais modesto.
    modelo = YOLO('yolov26n.pt') 

    print(f"Iniciando treinamento com o dataset {caminho_yaml} por {epocas} épocas...")
    # O device='' deixa a biblioteca escolher automaticamente (GPU se disponível, senão CPU)
    resultados = modelo.train(data=caminho_yaml, epochs=epocas, imgsz=tamanho_img, device='')
    
    print("Treinamento finalizado! Os pesos do modelo foram salvos na pasta 'runs/detect/train/weights/'")
    return resultados


def detectar_em_imagem(caminho_modelo: str, caminho_imagem: str):
    """
    Carrega um modelo treinado e realiza a detecção em uma única imagem.
    
    Args:
        caminho_modelo (str): Caminho para o arquivo .pt (ex: 'best.pt').
        caminho_imagem (str): Caminho para a imagem que será analisada.
    """
    modelo = YOLO(caminho_modelo)
    
    # Realiza a predição (save=True salva o resultado visual na pasta runs/)
    resultados = modelo.predict(source=caminho_imagem, save=True, conf=0.5)
    
    print(f"Detecção concluída. Verifique a pasta 'runs/detect/predict' para ver o resultado.")


def detectar_em_webcam(caminho_modelo: str):
    """
    Realiza a detecção de objetos em tempo real usando a webcam.
    Pressione 'q' para fechar a janela.
    
    Args:
        caminho_modelo (str): Caminho para o arquivo .pt treinado.
    """
    modelo = YOLO(caminho_modelo)
    
    # Inicia a captura de vídeo (0 geralmente é a webcam principal)
    captura = cv2.VideoCapture(0)
    
    if not captura.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    print("Iniciando detecção em tempo real... Pressione 'q' para sair.")
    
    while True:
        sucesso, frame = captura.read()
        if not sucesso:
            break
            
        # Roda a predição no frame atual
        resultados = modelo(frame, conf=0.5)
        
        # O YOLO já desenha as caixas delimitadoras pra gente
        frame_anotado = resultados[0].plot()
        
        # Mostra o resultado na tela
        cv2.imshow("Deteccao YOLO em Tempo Real", frame_anotado)
        
        # Condição de saída: tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    captura.release()
    cv2.destroyAllWindows()


# =====================================================================
# BLOCO DE EXECUÇÃO
# Descomente a função que você deseja rodar.
# =====================================================================
if __name__ == "__main__":
    
    # 1. PARA TREINAR (precisa do arquivo data.yaml e das imagens configuradas):
    # treinar_modelo("meu_dataset/data.yaml", epocas=30)
    
    # 2. PARA TESTAR EM UMA IMAGEM (usando o modelo nano pré-treinado do YOLO):
    # detectar_em_imagem("yolov8n.pt", "caminho_para_sua_foto.jpg")
    
    # 3. PARA TESTAR NA WEBCAM (usando o modelo nano pré-treinado):
    # detectar_em_webcam("yolov8n.pt")
    pass