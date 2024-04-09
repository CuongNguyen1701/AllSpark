import os
import random
dataset_name = "easyportrait"
dataset_path = "EasyPotrait"
ann_path = os.path.join(dataset_path, "annotations")
img_path = os.path.join(dataset_path, "images")
split = 4  # unlabeled data is 4 times the labeled data
os.makedirs(f"splits/{dataset_name}/1_{split}", exist_ok=True)

train_ann_path = os.path.join(ann_path, "train")
train_img_path = os.path.join(img_path, "train")
val_ann_path = os.path.join(ann_path, "val")
val_img_path = os.path.join(img_path, "val")

train_ids = []

# # generate labeled and unlabeled data for training
# for f in os.listdir(train_ann_path):
#     if not f.endswith(".png"):
#         continue
#     id = os.path.splitext(f)[0]
#     train_ids.append(id)

# for id in train_ids:
#     if (os.path.exists(os.path.join(train_img_path, id+".jpg")) and os.path.exists(os.path.join(train_ann_path, id+".png"))):
#         # write image path and annotation path to the labeled.txt and unlabeled.txt
#         if(random.random() < 1/(split+1)):
#             with open(f"splits/{dataset_name}/1_{split}/labeled.txt", "a") as f:
#                 f.write(f"{os.path.join(train_img_path, id+'.jpg')} {os.path.join(train_ann_path, id+'.png')}\n")
#         else:
#             with open(f"splits/{dataset_name}/1_{split}/unlabeled.txt", "a") as f:
#                 f.write(f"{os.path.join(train_img_path, id+'.jpg')} {os.path.join(train_ann_path, id+'.png')}\n")

val_ids = []
for f in os.listdir(val_ann_path):
    if not f.endswith(".png"):
        continue
    id = os.path.splitext(f)[0]
    val_ids.append(id)

for id in val_ids:
    if (os.path.exists(os.path.join(val_img_path, id+".jpg")) and os.path.exists(os.path.join(val_ann_path, id+".png"))):
        with open(f"splits/{dataset_name}/val.txt", "a") as f:
            f.write(
                f"{os.path.join(val_img_path, id+'.jpg')} {os.path.join(val_ann_path, id+'.png')}\n")
