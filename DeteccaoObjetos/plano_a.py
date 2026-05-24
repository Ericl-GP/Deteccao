import cv2
from ultralytics import YOLO

def verificar_intersecao(caixa1, caixa2):
    """
    Verifica se duas caixas delimitadoras estão se tocando ou sobrepostas.
    Formato da caixa: [x1, y1, x2, y2] (coordenadas dos cantos superior esquerdo e inferior direito)
    """
    x1_1, y1_1, x2_1, y2_1 = caixa1
    x1_2, y1_2, x2_2, y2_2 = caixa2
    
    # Se uma caixa está totalmente à esquerda, à direita, acima ou abaixo da outra, não há interseção
    if x1_1 > x2_2 or x2_1 < x1_2 or y1_1 > y2_2 or y2_1 < y1_2:
        return False
    return True

def executar_plano_a():
    print("Carregando modelo YOLO pré-treinado...")
    modelo = YOLO('yolov8n.pt')
    
    # Inicia a webcam (0). Se quiser testar um vídeo gravado, troque 0 pelo caminho do vídeo (ex: 'video_escola.mp4')
    captura = cv2.VideoCapture(0)
    
    if not captura.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return

    print("Sistema rodando! Pressione 'q' para sair.")

    while True:
        sucesso, frame = captura.read()
        if not sucesso:
            break
            
        # Roda o YOLO no frame atual (verbose=False oculta os logs no terminal para não poluir)
        resultados = modelo(frame, conf=0.5, verbose=False)
        caixas_yolo = resultados[0].boxes
        
        pessoas = []
        bolsas = []
        
        # 1. Separar as detecções em listas
        for caixa in caixas_yolo:
            classe_id = int(caixa.cls[0])
            coords = caixa.xyxy[0].tolist() # Pega as coordenadas [x1, y1, x2, y2]
            
            if classe_id == 0:
                pessoas.append(coords)
            elif classe_id in [1, 2]: # 1 = backpack (mochila), 2 = handbag (bolsa de lado)
                bolsas.append(coords)
                
                # Opcional: desenhar a caixinha menor da mochila em azul para depuração
                bx1, by1, bx2, by2 = map(int, coords)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 1)
                
        # 2. Aplicar a lógica Aluno vs Professor
        for p_coords in pessoas:
            px1, py1, px2, py2 = map(int, p_coords)
            eh_aluno = False
            
            # Compara a pessoa atual com todas as bolsas detectadas na tela
            for b_coords in bolsas:
                if verificar_intersecao(p_coords, b_coords):
                    eh_aluno = True
                    break # Se achou uma bolsa colada na pessoa, já confirma que é aluno
            
            # 3. Desenhar o resultado final
            if eh_aluno:
                cor = (0, 255, 0) # Verde (BGR no OpenCV)
                texto = "Aluno"
            else:
                cor = (0, 0, 255) # Vermelho
                texto = "Pessoa"
                
            # Desenha a caixa ao redor da pessoa
            cv2.rectangle(frame, (px1, py1), (px2, py2), cor, 2)
            
            # Adiciona o texto acima da caixa
            cv2.putText(frame, texto, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        # Mostra o vídeo com as anotações
        cv2.imshow("Monitoramento Escolar - Plano A", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    captura.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    executar_plano_a()