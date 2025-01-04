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


def create_folder_structure(base_path, classes):
    """Создает структуру папок для датасета."""
    for cls in classes:
        os.makedirs(os.path.join(base_path, 'train', cls), exist_ok=True)


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
