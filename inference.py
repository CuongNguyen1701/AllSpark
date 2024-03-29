import argparse
import torch
import yaml
from model.model_helper import ModelBuilder
from dataset.semi import SemiDataset
from torch.utils.data import DataLoader
import numpy as np
import os
parser = argparse.ArgumentParser(description='Inference script for the model')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--model-path', type=str, required=True)
parser.add_argument('--data-path', type=str, required=True)
parser.add_argument('--output-path', type=str, required=True)

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    model = ModelBuilder(cfg['model'])

    # Load the trained model
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Load the data
    dataset = SemiDataset(cfg['dataset'], args.data_path, 'test', cfg['crop_size'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Make predictions on the data
    for i, (img, _) in enumerate(dataloader):
        img = img.cuda()
        with torch.no_grad():
            output = model(img)
        pred = output.argmax(dim=1).cpu().numpy()

        # Save the predictions
        np.save(os.path.join(args.output_path, f'pred_{i}.npy'), pred)

if __name__ == '__main__':
    main()