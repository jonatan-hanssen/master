import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
from tqdm import tqdm
from torchvision.models import resnet50, vgg16


from utils import CustomDataset, save_features
from utils import *

device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--generate", "-g", action="store_true")
args = parser.parse_args()

model = resnet50(weights="DEFAULT")
model.to(device)


# root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")

id_feature_train_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "id_features_train.pt")
id_feature_val_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "id_features_val.pt")
ood_feature_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ood_features.pt")

imageneto_feature_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imageneto_features.pt")

# id_lrp_train_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "id_features_train.pt")
# id_lrp_val_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "id_features_val.pt")
# ood_lrp_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ood_features.pt")


id_gradcam_train_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "id_gradcam_train.pt")
id_gradcam_val_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "id_gradcam_val.pt")
ood_gradcam_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ood_gradcam.pt")

imageneto_gradcam_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imageneto_gradcam.pt")

# Instantiate the dataset
id_dataset = CustomDataset(root_dir=root, path="imagenet-val")
ood_dataset = CustomDataset(root_dir=root, path="iNaturalist/images")
imageneto_dataset = CustomDataset(root_dir=root, path="imagenet-o")

id_dataset_train, id_dataset_val = random_split(
    id_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(0)
)



id_dataloader_train = DataLoader(id_dataset_train, batch_size=32)
id_dataloader_val = DataLoader(id_dataset_val, batch_size=32)
ood_dataloader = DataLoader(ood_dataset, batch_size=32)

imageneto_dataloader = DataLoader(imageneto_dataset, batch_size=32)

if args.generate:
    # save_features(model, id_dataloader_train, id_feature_train_save_path)
    # save_features(model, id_dataloader_val, id_feature_val_save_path)
    # save_features(model, ood_dataloader, ood_feature_save_path)

    # save_gradcams(model, id_dataloader_train, id_gradcam_train_save_path)
    # save_gradcams(model, id_dataloader_val, id_gradcam_val_save_path)
    # save_gradcams(model, ood_dataloader, ood_gradcam_save_path)

    save_features(model, imageneto_dataloader, imageneto_feature_save_path)
    save_gradcams(model, imageneto_dataloader, imageneto_gradcam_save_path)
    exit()


feature_id_train = torch.load(id_feature_train_save_path, map_location="cpu")
feature_id_val = torch.load(id_feature_val_save_path, map_location="cpu")
feature_ood = torch.load(ood_feature_save_path, map_location="cpu")

feature_imageneto = torch.load(imageneto_feature_save_path, map_location="cpu")

# vim(model, feature_id_train, feature_id_val, feature_ood)

gradcam_id_train = torch.load(id_gradcam_train_save_path, map_location="cpu")
gradcam_id_val = torch.load(id_gradcam_val_save_path, map_location="cpu")
gradcam_ood = torch.load(ood_gradcam_save_path, map_location="cpu")

gradcam_imageneto = torch.load(imageneto_gradcam_save_path, map_location="cpu")

gradcam_id_train = gradcam_id_train.mean(dim=0).reshape(-1, 7, 7).squeeze()
gradcam_ood = gradcam_ood.mean(dim=0).reshape(-1, 7, 7).squeeze()

plt.subplot(121)
plt.imshow(gradcam_id_train, cmap="hot", interpolation="nearest")
plt.subplot(122)
plt.imshow(gradcam_ood, cmap="hot", interpolation="nearest")
plt.show()
print(gradcam_id_train.shape)



# print("neovim")
# neovim(model, feature_id_train, feature_id_val, feature_ood, gradcam_id_train, gradcam_id_val, gradcam_ood)
#
# print("vim")
# vim(model, feature_id_train, feature_id_val, feature_ood)
#
# print("neovim")
# neovim(model, feature_id_train, feature_id_val, feature_imageneto, gradcam_id_train, gradcam_id_val, gradcam_imageneto)
#
# print("vim")
# vim(model, feature_id_train, feature_id_val, feature_imageneto)

