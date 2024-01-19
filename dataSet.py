import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class Datasets(Dataset):
    def __init__(self, train_dir, val_dir, mode="train"):
        self.transform = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }[mode]

        self.mode = mode

        if self.mode == "train":
            self.main_dir = train_dir
        else:
            self.main_dir = val_dir

        self.all_imgs = []
        self.all_labels = []
        self.class_labels = {}

        for label, class_folder in enumerate(os.listdir(self.main_dir)):
            self.class_labels[label] = class_folder
            class_folder_path = os.path.join(self.main_dir, class_folder)

            for file in os.listdir(class_folder_path):
                if file.endswith(".jpg") or file.endswith(".png"):  # checks if the file is an image
                    self.all_imgs.append(os.path.join(class_folder_path, file))
                    self.all_labels.append(label)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = self.all_imgs[idx]
        tensor_image = None

        while tensor_image is None:
            try:
                name=os.path.basename(img_loc)
                image = Image.open(img_loc).convert("RGB")
                tensor_image = self.transform(image)
                label = self.all_labels[idx]

            except OSError as e:
                print(f"OSError: {e}")
                print(f"Image at {img_loc} could not be opened and will be skipped.")
                # code to remove the file
                try:
                    os.remove(img_loc)
                    print(f"Image at {img_loc} has been removed.")
                except Exception as rm_ex:
                    print(f"Could not remove image at {img_loc}. Reason: {rm_ex}")

                # increment idx and wrap around if necessary to get the next image
                idx = (idx + 1) % len(self.all_imgs)
                img_loc = self.all_imgs[idx]

        return tensor_image, label,name