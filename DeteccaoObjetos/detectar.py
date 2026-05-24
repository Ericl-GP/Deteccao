import cv2
from ultralytics import YOLO

def testar_webcam(caminho_pesos="runs/detect/train/weights/best.pt"):
    print(f"Carregando o modelo treinado: {caminho_pesos}")
    
    try:
        modelo = YOLO(caminho_pesos)
    except FileNotFoundError:
        print("Erro: Arquivo 'best.pt' não encontrado. Você precisa rodar o treinar.py primeiro!")
        return

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return

    print("Pressione 'q' para sair.")

    while True:
        sucesso, frame = captura.read()
        if not sucesso:
            break
            
        # Roda a detecção (conf=0.5 significa que só mostra se tiver 50%+ de certeza)
        resultados = modelo(frame, conf=0.5)
        
        # Desenha as caixas na imagem
        frame_anotado = resultados[0].plot()
        
        cv2.imshow("Teste do Modelo Treinado", frame_anotado)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    captura.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Quando for testar, basta rodar este arquivo
    testar_webcam()