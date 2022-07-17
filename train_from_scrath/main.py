from train import custom_dataset_folder, split_train_test
import torch, torchvision
import ssl
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context

    # check model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)

    dataset = custom_dataset_folder('./images_v3_classification')
    train_loader, val_loader = split_train_test(dataset, 0.7, 1, batch_size=96)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, betas=(0.9,0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model.to(device)

    print("Training start")

    for epoch in range(200):   # Set the number of epoch
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # [inputs, labels] are given by data loader
            inputs, labels = data[0].to(device), data[1].to(device)

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계를 출력합니다.
            running_loss += loss.item()
            if i % 20 == 19:    # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0
        print("lr: ", optimizer.param_groups[0]['lr'])

    print('Finished Training')

    PATH = './test_net2.pth'

    torch.save(model.state_dict(), PATH)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

