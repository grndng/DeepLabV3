# TODO: get code working so I can predict multiple image files and get IoU statistics for each epoch

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import os

def single_epoch_statistics(model_path: str, image_path: str, mask_path: str, pred_path: str, channels: int) -> int:
    model = torch.load(model_path)
    model.eval()
    
    for image_file, mask_file in zip(os.listdir(image_path),os.listdir(mask_path)):
        img = tifffile.imread(f"{image_path}/{image_file}")[:,:,:channels].transpose(2,0,1)
        #mask = tifffile.imread(f"{mask_path}/{mask_file}")

        # make prediction
        with torch.no_grad():
            pred = model(torch.from_numpy(img).unsqueeze(0).type(torch.cuda.FloatTensor)/255)
            plt.figure(figsize=(10,10))
            plt.imshow(pred['out'].cpu().detach().numpy()[0][0] > 0)
            
#            tifffile.imwrite(f"{pred_path}/{mask_file}", output_file)


#     flattened_output = pred['out'].data.cpu().numpy().flatten()
#     values, counts = np.unique(flattened_output, return_counts=True)

#     plt.hist(flattened_output)
#     plt.show()

#     img = tifffile.imread(image_path)
#     img = img[:,:,:3]

#     threshold = 0

# def plot_figure(image, mask, prediction, threshold):
#     plt.figure(figsize=(10,10))
#     plt.subplot(131)
#     plt.imshow(image)
#     plt.title('Image')
#     plt.axis('off')
#     plt.subplot(132)
#     plt.imshow(mask)
#     plt.title('Ground Truth')
#     plt.axis('off')
#     plt.subplot(133)
#     plt.imshow(prediction['out'].cpu().detach().numpy()[0][0] > threshold) # wie wähl ich diesen Wert am besten
#     plt.title('Segmentation Output')
#     plt.axis('off')
#     plt.show()

# plot_figure(img, mask, pred, threshold)

single_epoch_statistics(
    model_path = r"D:\Potsdam_Final\Training_Results\Potsdam_512_H\weights.pt",
    image_path = r"D:\Potsdam_Final\512\Images",
    mask_path = r"D:\Potsdam_Final\512\Masks",
    pred_path = r"D:\Potsdam_Final\Training_Results\Potsdam_512_H\predictions",
    channels = 1
)