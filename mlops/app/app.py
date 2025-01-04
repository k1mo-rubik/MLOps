import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Загрузка модели ONNX
onnx_model_path = "trained_model.onnx"
session = ort.InferenceSession(onnx_model_path)

# Получение входных и выходных имен модели
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Классы модели (замените на ваши классы)
classes = ['автобус', 'автомобиль', 'грузовик', 'мотоцикл']


# Функция для предобработки изображения
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image.numpy()


# Функция для предсказания
def predict(image):
    # Предобработка изображения
    input_data = preprocess_image(image)

    # Инференс модели
    predictions = session.run([output_name], {input_name: input_data})[0]

    # Получение предсказанного класса
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Возвращаем имя класса
    return classes[predicted_class]


# Интерфейс Gradio
def classify_image(image):
    label = predict(image)
    return f"Предсказанный класс: {label}"


# Настройка интерфейса
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Классификация изображений",
    description="Загрузите изображение, чтобы узнать его класс"
)

if __name__ == "__main__":
    # Запуск приложения
    interface.launch(server_name="0.0.0.0", server_port=7860)
