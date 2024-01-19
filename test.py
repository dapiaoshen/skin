from dataSet import Datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from train import Trainer
import argparse
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--model_name', type=str, default="mobile_v3_large")
    parser.add_argument('--Model_Path', type=str, default="/home/lixinyu/skindata/skin/mobile_55.pth")
    parser.add_argument('--train_dir', type=str, default="/home/lixinyu/skindata/skin/train")
    parser.add_argument('--val_dir', type=str, default="/home/lixinyu/skindata/skin/val")
    parser.add_argument('--test_dir', type=str, default="/home/lixinyu/skindata/skin/test")
    parser.add_argument('--loss_dir', type=str, default="./losses/")
    parser.add_argument('--save_model_dir', type=str, default="/home/lixinyu/skindata/skin/model")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--flags', type=str, default="train")
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--class_num', type=int, default=29)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--frozenNums', type=int, default=0)
    parser.add_argument('--gpu', default='0', type=str)

    args = parser.parse_known_args()[0]
    # torch.cuda.set_device(args.local_rank)
    # 选择第二块GPU进行训练
    selected_gpu = int(args.gpu)  # 将字符串转换为整数

    device = f"cuda:{selected_gpu}"

    dataSet_train = Datasets(train_dir=args.train_dir, val_dir=args.val_dir, mode="train")
    dataSet_val = Datasets(train_dir=args.val_dir, val_dir=args.val_dir, mode="val")
    dataSet_test = Datasets(train_dir=args.test_dir, val_dir=args.val_dir, mode="val")
    train_loader = data.DataLoader(dataSet_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                   num_workers=1)
    val_loader = data.DataLoader(dataSet_val, batch_size=1, shuffle=False, drop_last=True, num_workers=1)
    test_loader=data.DataLoader(dataSet_test, batch_size=1, shuffle=False, drop_last=True, num_workers=1)
    trainer = Trainer(data_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      max_epoch=args.max_epoch,
                      save_path=args.save_model_dir,
                      device=device,
                      class_num=args.class_num,
                      lr=args.lr,
                      pretrained=args.pretrained,
                      model_path=args.Model_Path,
                      model_name=args.model_name
                      )

    test_loss, test_acc = trainer.evaluate(test_loader,args.Model_Path)
