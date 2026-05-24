from ultralytics import YOLO

def treinar_modelo_ifto():
    print("Carregando o YOLO26 Nano...")
    modelo = YOLO('yolo26n.pt') 

    print("Iniciando treinamento com os dados do IFTO...")
    resultados = modelo.train(
        data='dataset.yaml',   # O arquivo gerado pelo software de anotação
        epochs=100,            # 100 épocas é um bom número para modelos customizados
        imgsz=640,             
        batch=8,               # Tamanho do lote seguro para sua RAM e RX 580
        device='',             
        plots=True,
        name='modelo_ifto_v1'  # Nome da pasta onde os pesos serão salvos
    )
    print("Treinamento finalizado!")

if __name__ == '__main__':
    treinar_modelo_ifto()