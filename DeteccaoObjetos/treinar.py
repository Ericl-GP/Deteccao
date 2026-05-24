from ultralytics import YOLO

def main():
    print("Carregando o modelo base YOLO26 Nano...")
    modelo = YOLO('yolo26n.pt') 

    print("Iniciando o treinamento...")
    # O YOLO salva automaticamente o melhor modelo em 'runs/detect/train/weights/best.pt'
    resultados = modelo.train(
        data='dataset.yaml',   # Aponta para o seu arquivo de configuração
        epochs=50,             # Número de ciclos de treinamento
        imgsz=640,             # Resolução padrão do YOLO
        batch=16,              # Imagens processadas por vez (16 ou 8 são bons para seus 16GB de RAM)
        device='',             # Deixe vazio para ele tentar usar GPU, ou force 'cpu' se o ROCm falhar
        plots=True             # Gera gráficos de desempenho no final
    )
    print("Treinamento finalizado!")

if __name__ == '__main__':
    main()