import torch, os
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import argparse
import matplotlib

matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision

from torchvision.models import resnet18

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.nn import Linear, Sequential, Sigmoid, Softmax

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
house_dataset_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "house_data"
)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")


class CatDogDataset(Dataset):
    def __init__(self, path, use_outliers=False):
        self.path = path

        if use_outliers:
            outlier_dirs = os.listdir(os.path.join(self.path, "outliers"))

            outliers = list()

            for outlier_dir in outlier_dirs:
                files = os.listdir(os.path.join(self.path, "outliers", outlier_dir))

                files = [os.path.join(self.path, "outliers", outlier_dir, file) for file in files]


                outliers += files

            outliers = [(outlier, 0) for outlier in outliers]
            self.data = outliers
            return



        cats = os.listdir(os.path.join(self.path, "cat"))
        cats = [(os.path.join(self.path, "cat", cat), 1) for cat in cats]

        dog_breed_dirs = os.listdir(os.path.join(self.path, "dog"))

        dogs = list()

        for dog_breed_dir in dog_breed_dirs:
            files = os.listdir(os.path.join(self.path, "dog", dog_breed_dir))

            files = [os.path.join(self.path, "dog", dog_breed_dir, file) for file in files]
            dogs += files


        dogs = [(dog, 0) for dog in dogs]

        self.data = cats + dogs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, is_cat = self.data[idx]

        image = Image.open(img_path)

        if image.mode != "RGB":
            image = image.convert("RGB")

        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image_tensor = preprocess(image)

        truth_tensor = torch.tensor([1, 0]) if is_cat else torch.tensor([0, 1])

        return image_tensor, truth_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action="store_true", help="Test")
    parser.add_argument("-o", "--outliers", action="store_true", help="Use outliers dataset")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"


    dataset = CatDogDataset(dataset_path, use_outliers=args.outliers)

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    if not args.test:
        model = resnet18(weights="IMAGENET1K_V1")
        model.fc = Sequential(
            Linear(in_features = 512, out_features=2, bias=True),
            Softmax(dim=1)
        )

        model.to(device)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        progress_bar = tqdm(dataloader)

        # training
        for i, data in enumerate(progress_bar):
            batch, labels = data

            batch = batch.to(device)
            labels = labels.to(device)
            labels = labels.float()

            pred = model(batch)
            pred = pred.squeeze()
            loss = loss_fn(pred, labels)

            progress_bar.set_description(f"Loss: {loss:.5f}")

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if (i + 1) % 200 == 0:
                print("saving model")
                torch.save(model.state_dict(), model_path)

    else:
        model = resnet18()
        model.fc = Sequential(
            Linear(in_features = 512, out_features=2, bias=True),
            Softmax(dim=1)
        )

        model.load_state_dict(torch.load(model_path))
        model.to(device)

        # print(model)

        target_layers = [model.layer4[-1]]

        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

        for i, data in enumerate(dataloader):
            batch, labels = data

            batch = batch.to(device)
            labels = labels.to(device)
            labels = labels.float()

            grayscale_cam = cam(input_tensor=batch, aug_smooth=True, eigen_smooth=True)
            preds = model(batch)
            preds = preds.squeeze()

            i = 0
            for image, pred, cam_img in zip(batch, preds, grayscale_cam):
                i += 1
                # image = image.to("cpu")


                unnormalize = transforms.Compose(
                    [
                        transforms.Normalize(
                            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                        ),
                        transforms.Normalize(
                            mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]
                        ),
                    ]
                )
                image = unnormalize(image)

                visualization = show_cam_on_image(
                    np.float32(torchvision.transforms.ToPILImage()(image)) / 255,
                    cam_img,
                    use_rgb=True,
                    image_weight=0.6,
                )

                plt.subplot(130 + i)
                plt.imshow(torch.einsum("chw->hwc", image.to("cpu")))
                # plt.subplot(212)
                plt.imshow(visualization)
                plt.title(f"cat: {pred[0]:.2f}, dog: {pred[1]:.2f}")
                if i == 3:
                    plt.show()
                    i = 0

            break
