import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from duckduckgo_search import DDGS
import requests
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image


def download_images(class_name, output_folder, num_images):
    """Скачивает изображения для указанного класса."""
    with DDGS() as ddgs:
        ddgs_images_gen = ddgs.images(
            class_name,
            region="wt-wt",
            size="Medium",
            type_image="photo",
            max_results=num_images
        )

        for idx, image_data in enumerate(tqdm(ddgs_images_gen, total=num_images, desc=f"Downloading {class_name}")):
            image_url = image_data['image']
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()

                # Сохраняем изображение
                image_path = os.path.join(output_folder, f"{class_name}_{idx + 1}.jpg")
                with open(image_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Failed to download {image_url}: {e}")


# # Параметры
# classes = ["автомобиль", "мотоцикл", "автобус", "грузовик"]
# base_path = "dataset"
# num_images_per_class = 150
#
# # Скачиваем изображения
# for cls in classes:
#     train_folder = os.path.join(base_path, 'train', cls)
#     download_images(cls, train_folder, num_images_per_class)


def clean_dataset(dataset_path):
    """Удаляет поврежденные изображения из указанного пути."""
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Проверяет, можно ли открыть файл
            except (IOError, SyntaxError):
                print(f"Удаление поврежденного файла: {file_path}")
                os.remove(file_path)


def create_folder_structure(base_path, classes):
    """Создает структуру папок для датасета."""
    for cls in classes:
        os.makedirs(os.path.join(base_path, 'train', cls), exist_ok=True)


def get_transforms_pipeline():
    """
    Создает преобразования данных для обучения и тестирования.
    """
    # Определение трансформаций для тестовых данных
    transforms_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transforms_pipeline


def get_datasets(base_path, transforms_pipeline):
    """
    Загружает датасеты для обучения и тестирования.
    """
    # Создаем Dataset
    full_dataset = ImageFolder(root=os.path.join(base_path, 'train'), transform=transforms_pipeline)

    # Разделение на train и test (80% / 20%)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    return train_dataset, test_dataset


def get_dataloaders(image_datasets, batch_size=32):
    """
    Создает DataLoader для обучения и тестирования.

    Args:
        image_datasets (dict): Датасеты для train и val.
        batch_size (int, optional): Размер батча. По умолчанию 32.

    Returns:
        dict: Словарь с DataLoader для train и val.
    """
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4),
    }
    return dataloaders


def get_loaders(train_dataset, test_dataset):
    # Создаем DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return train_loader, test_loader