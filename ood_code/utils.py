import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance

from scipy.special import logsumexp, softmax

from numpy.linalg import norm, pinv


def save_features(model, loader, save_path):
    features = list()

    def hook(model, input, output):
        output = output.squeeze(-1).squeeze(-1)
        features.append(output.detach())

    h1 = model.avgpool.register_forward_hook(hook)

    for i, data in enumerate(tqdm(loader)):
        batch, = data["image"].to(device)


        model.eval()
        with torch.no_grad():
            model(batch)

    h1.remove()

    features = torch.cat(features)

    torch.save(features, save_path)


class CustomDataset(Dataset):
    def __init__(self, root_dir, path, transform=None):
        self.root_dir = root_dir

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transform

        self.data = list()  # list of tuples (image_file, label)

        dir_path = os.path.join(root_dir, path)

        for img_file in os.listdir(dir_path):
            self.data.append(os.path.join(dir_path, img_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        image = Image.open(img_path)  # Using PIL to open image

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform(image)

        return image


def save_features(model, loader, save_location, device="cuda"):
    features = list()

    hook_func = lambda model, input, output : features.append(output.squeeze().detach())

    hook = model.avgpool.register_forward_hook(hook_func)

    for i, data in enumerate(tqdm(loader)):
        batch = data.to(device)

        batch = batch.to(device)


        model.eval()
        with torch.no_grad():
            model(batch)

    hook.remove()

    features = torch.cat(features)
    print(features.shape)
    print(features.device)

    torch.save(features, save_location)

def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh

def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.0
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh


def vim(model, feature_id_train, feature_id_val, feature_ood):
    w = model.fc.weight.data.cpu().numpy()
    b = model.fc.bias.data.cpu().numpy()

    logit_id_train = feature_id_train @ w.T + b
    logit_id_val = feature_id_val @ w.T + b
    logit_ood = feature_ood @ w.T + b

    logit_id_train = np.array(logit_id_train)
    logit_id_val = np.array(logit_id_val)
    logit_ood = np.array(logit_ood)

    # print(logit_id_train)
    # print(logit_id_val)
    # print(logit_ood)


    # origin point with moore penrose inverse
    u = - (pinv(w) @ b)

    DIM = 256

    print("computing principal space...")
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)

    null_space = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    print("computing alpha...")
    vlogit_id_train = norm(np.matmul(feature_id_train - u, null_space), axis=-1)


    # print(vlogit_id_train)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    print(f"{alpha=:.4f}")

    vlogit_id_val = norm(np.matmul(feature_id_val - u, null_space), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val

    energy_ood = logsumexp(logit_ood, axis=-1)
    vlogit_ood = norm(np.matmul(feature_ood - u, null_space), axis=-1) * alpha
    score_ood = -vlogit_ood + energy_ood
    fpr_ood, _ = fpr_recall(score_id, score_ood, 0.95)

    print(f"FPR95: {fpr_ood:.2%}")



