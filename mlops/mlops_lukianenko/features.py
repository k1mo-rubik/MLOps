import os
from torch import onnx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from duckduckgo_search import DDGS
import requests

from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image


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


def get_loaders(train_dataset, test_dataset):
    # Создаем DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return train_loader, test_loader


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


def export_to_onnx(model, input, output_path="model.onnx"):
    """
    Экспортирует модель в формат ONNX.

    Args:
        model (torch.nn.Module): Обученная модель.
        input (torch.Tensor): Пример входных данных для экспорта.
        output_path (str): Путь для сохранения ONNX модели.
    """
    model.eval()  # Переводим модель в режим оценки
    onnx.export(
        model,
        input,
        output_path,
        export_params=True,  # Экспортировать параметры модели
        opset_version=11,  # Версия ONNX opset
        do_constant_folding=True,  # Оптимизация
        input_names=['input'],  # Имена входов
        output_names=['output'],  # Имена выходов
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Динамические оси
    )
    print(f"Модель экспортирована в ONNX: {output_path}")