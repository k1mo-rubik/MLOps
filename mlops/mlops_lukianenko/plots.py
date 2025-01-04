from random import random
import matplotlib.pyplot as plt


def show_random_classes(train_dataset, n=3):
    # Создаем фигуру для отображения
    fig, axes = plt.subplots(n, n, figsize=(10, 10))

    # Отображаем 9 изображений
    for i, ax in enumerate(axes.flatten()):
        # Извлекаем изображение и класс
        img, label = train_dataset[i]  # используем индексы для выборки данных

        # Отображаем изображение
        ax.imshow(img.permute(1, 2, 0))  # Переставляем оси для отображения
        ax.axis('off')  # Отключаем оси

        # Подпись с названием класса (используем dataset.classes для получения классов)
        ax.set_title(train_dataset.dataset.classes[label])

    # Отображаем все изображения
    plt.tight_layout()
    plt.show()
