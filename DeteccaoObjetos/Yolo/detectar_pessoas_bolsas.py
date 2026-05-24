import cv2
from ultralytics import YOLO

def iniciar_deteccao_otimizada():
    print("Carregando o modelo YOLO pré-treinado...")
    modelo = YOLO('yolo26m.pt') # Modelo mais leve e rápido, mas ainda eficaz para objetos pequenos como mochilas
    
    # Nome do seu arquivo de vídeo
    fonte_video = "mochilas.mp4"
    
    print(f"Abrindo o arquivo de vídeo: {fonte_video} ...")
    captura = cv2.VideoCapture(fonte_video)
    
    if not captura.isOpened():
        print("[ERRO] Não foi possível abrir o arquivo de vídeo.")
        return

    print("Sistema rodando com otimização para objetos pequenos! Pressione 'q' para sair.")

    while True:
        sucesso, frame = captura.read()
        if not sucesso or frame is None:
            print("Fim do vídeo ou falha ao ler o frame.")
            break
            
        # ======================================================================
        # TUNING DO MODELO:
        # conf=0.22 -> Aceita detecções mais "escondidas" ou difíceis de bolsa
        # imgsz=960  -> Aumenta a visão do YOLO para capturar objetos menores (mochilas)
        # ======================================================================
        resultados = modelo(frame, stream=True, conf=0.22, imgsz=800, verbose=False)
        
        for resultado in resultados:
            caixas = resultado.boxes
            
            for caixa in caixas:
                classe_id = int(caixa.cls[0])
                confianca = float(caixa.conf[0]) # Pega a porcentagem de certeza da IA
                x1, y1, x2, y2 = map(int, caixa.xyxy[0])
                
                # Regra 1: Pessoa (ID 0)
                if classe_id == 0:
                    cor = (0, 255, 0) # Verde
                    # Mostra o nome e a certeza (ex: Pessoa 85%)
                    texto = f"Pessoa {confianca:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                    cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
                    
                # Regra 2: Mochila (24) ou Bolsa (26)
                elif classe_id in [24, 26]:
                    cor = (0, 0, 255) # Vermelho
                    texto = f"Bolsa {confianca:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                    cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        # Exibe o frame processado
        cv2.imshow("Monitoramento Otimizado - IFTO", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    captura.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    iniciar_deteccao_otimizada()