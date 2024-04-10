import os
import random
import argparse

parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--dataset-path', type=str, required=True)


def main():
    args = parser.parse_args()
    dataset_name = "easyportrait"
    dataset_path = args.dataset_path
    ann_path = os.path.join(dataset_path, "annotations")
    img_path = os.path.join(dataset_path, "images")
    split = 4  # unlabeled data is 4 times the labeled data
    os.makedirs(f"splits/{dataset_name}/1_{split}", exist_ok=True)

    train_ann_path = os.path.join(ann_path, "train")
    train_img_path = os.path.join(img_path, "train")
    val_ann_path = os.path.join(ann_path, "val")
    val_img_path = os.path.join(img_path, "val")

    train_ids = []

    # generate labeled and unlabeled data for training
    for f in os.listdir(train_ann_path):
        if not f.endswith(".png"):
            continue
        id = os.path.splitext(f)[0]
        train_ids.append(id)
    print(f"Total number of training images: {len(train_ids)}")
    labeled_count = 0
    unlabeled_count = 0
    for id in train_ids:
        if (os.path.exists(os.path.join(train_img_path, id+".jpg")) and os.path.exists(os.path.join(train_ann_path, id+".png"))):
            # write image path and annotation path to the labeled.txt and unlabeled.txt
            if (random.random() < 1/(split+1)):
                with open(f"splits/{dataset_name}/1_{split}/labeled.txt", "a") as f:
                    f.write(
                        f"{os.path.join(train_img_path, id+'.jpg')} {os.path.join(train_ann_path, id+'.png')}\n")
                labeled_count += 1
            else:
                with open(f"splits/{dataset_name}/1_{split}/unlabeled.txt", "a") as f:
                    f.write(
                        f"{os.path.join(train_img_path, id+'.jpg')} {os.path.join(train_ann_path, id+'.png')}\n")
                unlabeled_count += 1
    print(f"Number of labeled images: {labeled_count}")
    print(f"Number of unlabeled images: {unlabeled_count}")
    val_ids = []
    for f in os.listdir(val_ann_path):
        if not f.endswith(".png"):
            continue
        id = os.path.splitext(f)[0]
        val_ids.append(id)

    print(f"Number of validation images: {len(val_ids)}")
    for id in val_ids:
        if (os.path.exists(os.path.join(val_img_path, id+".jpg")) and os.path.exists(os.path.join(val_ann_path, id+".png"))):
            with open(f"splits/{dataset_name}/val.txt", "a") as f:
                f.write(
                    f"{os.path.join(val_img_path, id+'.jpg')} {os.path.join(val_ann_path, id+'.png')}\n")


if __name__ == '__main__':
    main()
