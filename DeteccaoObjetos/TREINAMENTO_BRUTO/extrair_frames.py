import cv2
import os
import glob

def extrair_frames(caminho_video, pasta_saida, intervalo_segundos=1):
    """
    Lê um vídeo e salva um frame a cada X segundos.
    """
    # Cria a pasta de saída se ela não existir
    os.makedirs(pasta_saida, exist_ok=True)
    
    captura = cv2.VideoCapture(caminho_video)
    
    if not captura.isOpened():
        print(f"Erro ao abrir o vídeo: {caminho_video}")
        return

    # Descobre o FPS real do vídeo (ex: 24, 30, 60)
    fps = round(captura.get(cv2.CAP_PROP_FPS))
    print(f"Vídeo detectado com {fps} FPS.")
    
    # Calcula quantos frames pular para dar 1 segundo
    intervalo_frames = fps * intervalo_segundos
    
    contador_frame = 0
    imagens_salvas = 0
    
    print(f"Iniciando extração para a pasta '{pasta_saida}'...")
    
    while True:
        sucesso, frame = captura.read()
        if not sucesso:
            break # Fim do vídeo
            
        # Salva apenas se o contador atingir o intervalo (ex: a cada 30 frames)
        if contador_frame % intervalo_frames == 0:
            nome_video = os.path.splitext(os.path.basename(caminho_video))[0]
            nome_arquivo = os.path.join(pasta_saida, f"{nome_video}_frame_{imagens_salvas:04d}.jpg")
            # Salva a imagem em alta qualidade
            cv2.imwrite(nome_arquivo, frame)
            imagens_salvas += 1
            
        contador_frame += 1
        
    captura.release()
    print(f"Extração concluída! {imagens_salvas} imagens foram salvas.")

if __name__ == "__main__":
    # Pasta com os vídeos
    PASTA_VIDEOS = "yolo_env\\TREINAMENTO_BRUTO\\exemplos"
    PASTA_DESTINO = "yolo_env\\TREINAMENTO_BRUTO\\dataset_bruto" # Pasta onde as imagens serão salvas
    
    # Encontra todos os arquivos .mp4 na pasta exemplos
    videos = glob.glob(os.path.join(PASTA_VIDEOS, "*.mp4"))
    
    if not videos:
        print(f"Nenhum arquivo .mp4 encontrado na pasta '{PASTA_VIDEOS}'.")
        exit()
    
    print(f"Encontrados {len(videos)} vídeos: {videos}")
    
    # Processa cada vídeo
    for video in videos:
        print(f"\nProcessando vídeo: {video}")
        # Extraindo 1 frame por segundo. 
        # Se quiser mais fotos, mude para 0.5 (2 fotos por segundo)
        extrair_frames(video, PASTA_DESTINO, intervalo_segundos=1)