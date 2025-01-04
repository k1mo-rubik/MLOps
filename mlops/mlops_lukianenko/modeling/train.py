import torch


def swap_head(pretrained_model, num_classes):
    pretrained_model.fc = torch.nn.Sequential(
        torch.nn.Linear(pretrained_model.fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, num_classes)
    )


def freeze_params(pretrained_model):
    for param in pretrained_model.parameters():
        param.requires_grad = False


def train_freezed(pretrained_model, num_epochs, train_loader, test_loader, device, loss_fn, optimizer, scheduler):
    train_mode_label = "Freezed training"
    train_process(num_epochs, pretrained_model, train_mode_label, train_loader, test_loader, device, loss_fn, optimizer,
                  scheduler)


def unfreeze(pretrained_model):
    # Разморозка всех параметров модели
    for param in pretrained_model.parameters():
        param.requires_grad = True


def train_fine_tune(num_epochs, pretrained_model, train_loader, test_loader, device, loss_fn, optimizer, scheduler):
    pretrained_model.train()
    train_mode_label = 'Fine-Tuning'

    train_process(num_epochs, pretrained_model, train_mode_label, train_loader, test_loader, device, loss_fn, optimizer,
                  scheduler)


def train_process(num_epochs, pretrained_model, train_mode_label, train_loader, test_loader, device, loss_fn, optimizer,
                  scheduler):
    for epoch in range(num_epochs):
        print(f"{train_mode_label} Epoch {epoch + 1}/{num_epochs}")
        pretrained_model.train()  # Устанавливаем режим тренировки
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = pretrained_model(inputs)
            loss = loss_fn(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"{train_mode_label} Training loss: {running_loss / len(train_loader)}")

        # Шаг изменения скорости обучения
        scheduler.step()

        # Оценка на тестовой выборке
        pretrained_model.eval()  # Устанавливаем режим оценки
        test_metrix(pretrained_model, test_loader, device)


def test_metrix(pretrained_model, test_loader, device):
    # Оценка на тестовой выборке
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = pretrained_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
