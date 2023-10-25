import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

def evaluate_with_model(image):
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)

# Inverter o dicionário de classes (mapeia índices inteiros para nomes de classes)
    inv_class_indices = {v: k for k, v in class_indices.items()}

# Carregar modelo
    model = load_model('turbidez_model.keras')

# Carregar a imagem
    image = cv2.imread('agua7.jpg')

# Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar desfoque Gaussiano para reduzir ruído e melhorar a detecção de bordas
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Usar detecção de bordas Canny
    edges = cv2.Canny(blurred, 50, 150)

# Encontrar contornos na imagem
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Inicializar variáveis para encontrar os extremos
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')

# Encontrar a caixa delimitadora que engloba todos os contornos
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

# Recortar a imagem usando as coordenadas extremas
    if x_max > x_min and y_max > y_min:
        cropped_image = image[y_min:y_max, x_min:x_max]

    # Redimensionar e preparar a imagem recortada para a predição
        img = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)).resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

   # Remove the batch dimension
        display_img = np.squeeze(img_array)

    # Converte de float para uint8 (Se necessário. O cv2 espera uint8)
        display_img = (display_img * 255).astype(np.uint8)

    # Fazendo a predição
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = inv_class_indices[predicted_index]

        return predicted_class

def calculate_mean_and_std(image, coord, size):
    x, y = coord
    half_size = size // 2
    h, w, _ = image.shape
    
    x1 = max(x - half_size, 0)
    x2 = min(x + half_size, w)
    y1 = max(y - half_size, 0)
    y2 = min(y + half_size, h)
    
    if x1 >= x2 or y1 >= y2:
        return None, None
    
    square_area = image[y1:y2, x1:x2]
    mean_color = np.mean(square_area, axis=(0, 1))
    std_color = np.std(square_area, axis=(0, 1))
    
    return mean_color, std_color

def evaluate_water(image, coord_copo, coord_fundo): 
    mean_copo, std_copo = calculate_mean_and_std(image, coord_copo, 20)
    mean_fundo, std_fundo = calculate_mean_and_std(image, coord_fundo, 20)

    color_difference = np.abs(mean_copo - mean_fundo)
    std_difference = np.abs(std_copo - std_fundo)

    weight = [0.3, 0.3, 0.4]
    weighted_difference = np.dot(color_difference, weight)
    
    # Calculando uma pontuação de turbidez que considera tanto a diferença de cores quanto o desvio padrão
    turbidity_score = weighted_difference + np.sum(std_difference)

    # Definindo limiares para classificação
    limiar_baixo = 35
    limiar_medio = 60

    print(turbidity_score)
    
    if turbidity_score <= limiar_baixo:
        print("Nível de turbidez: Baixo")
        return "Coloração com nível BAIXO de turbidez"
    elif turbidity_score <= limiar_medio:
        print("Nível de turbidez: Médio")
        return "Coloração com nível MÉDIO de turbidez"
    else:
        print("Nível de turbidez: Alto")
        return "Coloração com nível ALTO de turbidez"

@app.route('/evaluate_water', methods=['POST'])
def evaluate_water_endpoint():
    try:
        data = request.json
        image_base64 = data['image_base64']
        coord_copo = tuple(data['coord_copo'])
        coord_fundo = tuple(data['coord_fundo'])
        
        img_data = base64.b64decode(image_base64)
        image_np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(image_np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite('decoded_image.jpg', image)

        # Avaliar a qualidade da água usando o método antigo
        result1 = evaluate_water(image, coord_copo, coord_fundo)
        
        # Avaliar a qualidade da água usando o modelo de IA
        result2 = evaluate_with_model(image)

        return jsonify({'codigoAnaliseUm': result1, 'codigoAnaliseDois': result2})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
