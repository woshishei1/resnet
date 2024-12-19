import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from net import resnet34


# 超参数
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 数据集加载
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = resnet34(pretrained=False, num_classes=10).to(DEVICE)

loss_fn = nn.CrossEntropyLoss()

# 使用 SGD 优化器
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#torch.optim.RMSprop(lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9, eps=1e-8)
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
# 绘图数据
train_losses = []
train_accuracies = []
test_accuracies = []
test_precisions = []
test_recalls = []
test_losses = []


def plot_metrics():
    epochs = range(1, len(train_losses) + 1)

    # Directory to save the plot
    save_dir = 'plots'

    # Check if the directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12, 5))

    # Plot training loss and accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training Loss and Accuracy')
    plt.legend()

    # Plot test accuracy, precision, and recall
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.plot(epochs, test_precisions, label='Test Precision')
    plt.plot(epochs, test_recalls, label='Test Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Test Metrics')
    plt.legend()

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(save_dir, 'metrics_plot.png')
    plt.savefig(save_path)
    print(f"Plot saved at {save_path}")


def plot_metrics():
    # 确保保存目录存在

    epochs = range(1, len(train_losses) + 1)

    # 绘制训练损失和测试损失
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss and Test Loss')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "train_and_test_loss.png"))  # 保存独立的训练损失图

    # 绘制训练和测试准确率
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy and Train Accuracy')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "train_and_test_accuracy.png"))  # 保存独立的准确率图

    # 绘制测试精度
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, test_precisions, label='Test Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Test Precision')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "test_precision.png"))  # 保存独立的精度图

    # # Plot test recall
    # plt.subplot(2, 2, 4)  # 2 rows, 2 columns, fourth subplot
    # plt.plot(epochs, test_recalls, label='Test Recall')
    # plt.xlabel('Epoch')
    # plt.ylabel('Recall')
    # plt.title('Test Recall')
    # plt.savefig(os.path.join(SAVE_DIR, "Recall.png"))
    # plt.legend()
    # plt.show()
    # # 绘制测试准确率、精度和召回率
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, test_accuracies, label='Test Accuracy')
    # plt.plot(epochs, test_precisions, label='Test Precision')
    # plt.plot(epochs, test_recalls, label='Test Recall')
    # plt.xlabel('Epoch')
    # plt.ylabel('Metrics')
    # plt.title('Test Metrics')
    # plt.legend()


# 训练函数
def train(epoch, model, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_accuracy = 100. * correct / total
    print(f"Train Epoch: {epoch}\tLoss: {train_loss / len(train_loader):.6f}\tAccuracy: {train_accuracy:.2f}%")

    # 保存训练数据
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_accuracy)

    return train_loss / len(train_loader), train_accuracy

# 测试函数
def test(epoch, model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # 计算测试集的各项指标
    accuracy = (correct / total) * 100
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    # 保存测试数据
    test_accuracies.append(accuracy)
    test_precisions.append(precision)
    test_recalls.append(recall)
    test_losses.append(test_loss / len(test_loader))

    print(f"Test Epoch: {epoch}\tAccuracy: {accuracy:.4f}\tPrecision: {precision:.4f}\tRecall: {recall:.4f}")
    return accuracy, precision, recall


if __name__ == "__main__":
    NUM_EPOCHS = 2  # 设置训练轮数

    for epoch in range(1, NUM_EPOCHS + 1):
        train(epoch, model, optimizer)
        test(epoch, model)

    # 绘制训练和测试曲线
    plot_metrics()