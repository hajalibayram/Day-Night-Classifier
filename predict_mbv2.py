import os
import shutil
from glob import glob

import torch
from PIL import Image
from torchvision import models
import cv2
import argparse

from torchvision import transforms
from tqdm import tqdm


def load_model():
    # loading imagenet pretrained model from torchvision models
    mbv2 = models.mobilenet_v2(pretrained=True)
    in_features = mbv2.classifier[1].in_features
    # replacing final FC layer of pretrained model with our FC layer having output classes = 2 for day/night
    mbv2.classifier[1] = torch.nn.Linear(in_features, 2)
    # Load trained model onto CPU
    mbv2.load_state_dict(torch.load('models/mbv2_best_model.pth', map_location=torch.device('cpu')))
    # Setting model to evaluation mode
    mbv2.eval()
    return mbv2


def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img', required=False, help='Path to image')
    parser.add_argument('-f', '--folder', required=False, help='Path to image folder')
    parser.add_argument("--num_workers", type=int, default=12, help="_")
    parser.add_argument("--batch_size", type=int, default=16, help="_")
    parser.add_argument("--output_folder", type=str, default="outputs",
                        help="Folder where to save logs and maps")
    args = parser.parse_args()

    # imagenet mean and std to normalize data
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # loading model
    model = load_model()

    # reading image(s)
    images_folder = args.folder
    data_transforms = transforms.Compose([
        transforms.Resize((500, 500)),
        # transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, root=images_folder, transform=data_transforms):
            super().__init__()
            self.paths = sorted(glob(f"{root}/*"))
            import random
            random.shuffle(self.paths)
            self.transform = transform

        def __getitem__(self, index):
            path = self.paths[index]
            pil_image = Image.open(path).convert("RGB")
            return path, self.transform(pil_image), pil_image.size

        def __len__(self): return len(self.paths)

    image_dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    with torch.no_grad():
        for images_paths, images, images_sizes in tqdm(dataloader, ncols=80):
            outputs = model(images.to(device))
            outputs = torch.nn.functional.softmax(outputs, dim=1)

            print(images_paths)
            list_is_day = outputs[:, 0].tolist()
            for image_path, image, image_size, is_day in \
                    zip(images_paths, images, torch.stack(images_sizes).T, list_is_day):
                assert f"{is_day:f}"[0] == "0"
                print(is_day)
                image_name = os.path.basename(image_path)
                if is_day < 0.2:
                    _ = shutil.move(image_path, f"is_night/{image_name}")
                else:
                    _ = shutil.move(image_path, f"is_day/{image_name}")

if __name__ == '__main__':
    predict()
