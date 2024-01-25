import torch, os
from torchvision import transforms
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

from torchvision.models import resnet18

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.nn import Linear, Sequential, Sigmoid

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")

class CatDogaset(Dataset):

    def __init__(self, path):
        self.path = path
        cats = os.listdir(os.path.join(self.path, "cat"))
        dogs = os.listdir(os.path.join(self.path, "dog"))

        cats = [(cat, 1) for cat in cats]
        dogs = [(dog, 0) for dog in dogs]

        self.data = cats + dogs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name, is_cat = self.data[idx]

        if is_cat:
            img_path = os.path.join(self.path, "cat", img_name)
        else:
            img_path = os.path.join(self.path, "dog", img_name)

        image = Image.open(img_path)

        if image.mode != 'RGB':
            image = image.convert("RGB")

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image_tensor = preprocess(image)

        return image_tensor, is_cat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--test", action="store_true", help="Test"
    )

    args = parser.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CatDogaset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    if not args.test:
        model = resnet18(weights="IMAGENET1K_V1")
        model.fc = Sequential(
            Linear(in_features = 512, out_features=1, bias=True),
            Sigmoid()
        )

        model.to(device)

        loss_fn = torch.nn.BCELoss()
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
            Linear(in_features = 512, out_features=1, bias=True),
            Sigmoid()
        )

        model.load_state_dict(torch.load(model_path))
        model.to(device)

        for i, data in enumerate(dataloader):
            batch, labels = data

            batch = batch.to(device)
            labels = labels.to(device)
            labels = labels.float()


            preds = model(batch)
            preds = preds.squeeze()

            for image, pred in zip(batch, preds):
                image = image.to("cpu")
                plt.imshow(torch.einsum('chw->whc', image))
                plt.title(pred)
                plt.show()


            break

