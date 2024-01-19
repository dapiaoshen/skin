import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import torchvision.models as models
import torch.nn.functional as F
class Trainer:
    def __init__(self, data_loader, val_loader, test_loader, max_epoch, save_path, device, class_num, lr, pretrained, model_path, model_name):
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.max_epoch = max_epoch
        self.save_path = save_path
        self.device = device
        self.class_num = class_num
        self.lr = lr
        self.pretrained = pretrained
        self.model_path = model_path

        if model_name == "mobile_v3_large":
            self.model = models.mobilenet_v3_large(pretrained=pretrained)
            in_features = self.model.classifier[3].in_features
        elif model_name == "vgg":
            self.model = models.vgg16(pretrained=pretrained)
            in_features = self.model.classifier[0].in_features
        elif model_name == "resnet":
            self.model = models.resnet18(pretrained=pretrained)
            in_features = self.model.fc.in_features
        elif model_name == "densenet":
            self.model = models.densenet121(pretrained=pretrained)
            in_features = self.model.classifier.in_features
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # 通用的分类器结构
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, class_num)
        )

        self.model = self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train(self):
        losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        test_accs = []
        prediction=[]
        tar=[]
        for epoch in range(self.max_epoch):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets,name in tqdm(self.data_loader):
                inputs, targets= inputs.to(self.device), targets.to(self.device)


                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                # print(predicted)
                # print('_________')
                # print(targets)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                prediction += list(predicted.detach().cpu().numpy().flatten())
                tar+=list(targets.detach().cpu().numpy().flatten())

            print(prediction)
            print(tar)
            train_loss = running_loss / len(self.data_loader.dataset)
            train_acc = 100. * correct / total

            print(f'Epoch {epoch+1}/{self.max_epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')



        return losses, train_accs, val_losses, val_accs, test_accs

    def evaluate(self, data_loader,model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        skin_diseases = ["dermatomyositis", "telangiectasis", "flammeus", "atopic", "sd", "acne", "recurring", "gira",
                         "fungus", "solar", "glucocorticoids", "warts", "seborrheic", "contact", "angioedema",
                         "Keratosis", "tretinoin", "fdml", "drug", "perioral", "le", "psoriasis", "cellulitis",
                         "neurodermatitis", "aha", "erysipelas", "eczema", "makeup", "dirtadherent"]

        with torch.no_grad():
            for inputs, targets, names in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # 将输出转换为概率分布
                probabilities = F.softmax(outputs, dim=1)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                for name, probs in zip(names, probabilities):
                    disease_name = skin_diseases[predicted]
                    print(f"Image: {name}, Predicted Disease: {disease_name}")

                    # 打印概率大于10%的预测
                    for i, prob in enumerate(probs):
                        if prob > 0.1:  # 概率大于10%
                            print(f"  {skin_diseases[i]}: {prob.item() * 100:.2f}%")

        loss = running_loss / len(data_loader.dataset)
        acc = 100. * correct / total
        print('本次输出结果的准确度为')
        print(acc)
        return loss, acc