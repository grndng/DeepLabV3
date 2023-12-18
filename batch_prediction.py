# TODO: get code working so I can predict multiple image files and get IoU statistics for each epoch

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tifffile
import torch
import os

def single_epoch_predictions(model_path: str, image_path: str, mask_path: str, pred_path: str, channels: int) -> int:
    model = torch.load(model_path)
    model.eval()

    if not os.path.exists(pred_path):
        os.mkdir(pred_path)
    
    for image_file, mask_file in zip(os.listdir(image_path),os.listdir(mask_path)):
        img = tifffile.imread(f"{image_path}/{image_file}")[:,:,:channels].transpose(2,0,1)
        mask = tifffile.imread(f"{mask_path}/{mask_file}")

        # make prediction
        with torch.no_grad():
            pred = model(torch.from_numpy(img).unsqueeze(0).type(torch.cuda.FloatTensor))
            threshold = 0 #np.median(pred['out'].data.cpu().numpy().flatten())
            pred = np.invert(pred['out'].cpu().detach().numpy()[0][0] > threshold)
            tifffile.imwrite(f"{pred_path}/{image_file}", pred)


single_epoch_predictions(
    model_path = "D:/Potsdam_Final/Training_Results/Potsdam_512_RGBI/last_weights.pt",
    image_path = "D:/Potsdam_Final/512_test/Images",
    mask_path = "D:/Potsdam_Final/512_test/Masks",
    pred_path = "D:/Potsdam_Final/Training_Results/Potsdam_512_RGBI/predictions",
    channels = 4
    )