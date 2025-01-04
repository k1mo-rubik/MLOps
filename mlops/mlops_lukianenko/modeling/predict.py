import torch
from PIL import Image
from torchvision.transforms import transforms

import matplotlib.pyplot as plt


def predict(random_image_path, device, pretrained_model, train_dataset):
    # Загрузка изображения
    image = Image.open(random_image_path)
    # Определение трансформаций для тестовых данных

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Изменяем размер изображения
        transforms.ToTensor(),  # Преобразуем в тензор
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
    ])

    # Преобразование изображения
    transformed_image = test_transforms(image).unsqueeze(0)  # Добавляем размер батча

    # Перемещение на устройство (CPU/GPU)
    transformed_image = transformed_image.to(device)

    # Предсказание модели
    pretrained_model.eval()  # Устанавливаем режим оценки
    with torch.no_grad():
        output = pretrained_model(transformed_image)
        _, predicted_class = torch.max(output, 1)

    # Получение имени класса
    predicted_label = train_dataset.dataset.classes[predicted_class.item()]
    # Вывод изображения и предсказания
    plt.imshow(image)
    plt.title(f"Предсказание: {predicted_label}")
    plt.axis("off")
    plt.show()
