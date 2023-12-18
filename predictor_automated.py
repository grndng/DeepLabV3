import torch
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
import numpy as np
import os

def plot_figure(image, mask, prediction, threshold):
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Image')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(mask)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(prediction['out'].cpu().detach().numpy()[0][0] > threshold) # wie wähl ich diesen Wert am besten
    plt.title('Segmentation Output')
    plt.axis('off')
    plt.show()

model_path = f"D:/Potsdam_Final/Training_Results/Potsdam_512_RGBIH/last_weights.pt"
log_path = f"D:/Potsdam_Final/Training_Results/Potsdam_512_RGBIH/log.csv"

all_image_path = "D:/Potsdam_Final/512_test/Images/"
all_images = os.listdir(all_image_path)

model = torch.load(model_path)
model.eval()

for image in all_images:
    image_path = f"D:/Potsdam_Final/512_test/Images/{image}"
    mask_path = f"D:/Potsdam_Final/512_test/Masks/{image}"

    df_model = pd.read_csv(log_path)

#df_model.plot(x="epoch", figsize=(15,8))
#plt.show()

    print(df_model[['Train_auroc','Test_auroc']].max()) # was ist auroc

    img = tifffile.imread(image_path)[:,:,:].transpose(2,0,1)
    mask = tifffile.imread(mask_path)

    # make prediction
    with torch.no_grad():
        pred = model(torch.from_numpy(img).unsqueeze(0).type(torch.cuda.FloatTensor))

    flattened_output = pred['out'].data.cpu().numpy().flatten()
    values, counts = np.unique(flattened_output, return_counts=True)

    plt.hist(flattened_output)
    plt.show()

    img = tifffile.imread(image_path)
    img = img[:,:,:3]/255
    print(img.shape)

    threshold = 0
    plot_figure(img, mask, pred, threshold)
