import torch
import matplotlib.pyplot as plt
import tifffile
import pandas as pd
import numpy as np

channels = 5
model_path = f"./Training_{channels}ch/weights.pt"
log_path = f"./Training_{channels}ch/log.csv"

image_path = r"C:\Users\dinga\Documents\Doktorat\2023_CD_TrainData_cleansed\02_test_images\dop10rgbih_32359_5645_1_nw_tile8704_6144.tif"
mask_path = r"C:\Users\dinga\Documents\Doktorat\2023_CD_TrainData_cleansed\02_test_masks\dop10rgbih_32359_5645_1_nw_tile8704_6144.tif"

model = torch.load(model_path)
model.eval()

df_model = pd.read_csv(log_path)

#df_model.plot(x="epoch", figsize=(15,8))
#plt.show()

# print(df_model[['Train_auroc','Test_auroc']].max()) # was ist auroc

img = tifffile.imread(image_path)[:,:,:channels].transpose(2,0,1)
mask = tifffile.imread(mask_path)

# make prediction
with torch.no_grad():
    pred = model(torch.from_numpy(img).unsqueeze(0).type(torch.cuda.FloatTensor)/255)

flattened_output = pred['out'].data.cpu().numpy().flatten()
values, counts = np.unique(flattened_output, return_counts=True)

plt.hist(flattened_output)
plt.show()

img = tifffile.imread(image_path)
img = img[:,:,:3]

threshold = 0

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

plot_figure(img, mask, pred, threshold)