import torch
import matplotlib.pyplot as plt
import tifffile
import pandas as pd

model = torch.load("./Training_5ch/weights.pt")

image_path = "D:/Potsdam_Final/512/Images/Potsdam_tile0_1024.tif"
mask_path = "D:/Potsdam_Final/512/Masks/Potsdam_tile0_1024.tif"

img = tifffile.imread(image_path).transpose(2,0,1)
mask = tifffile.imread(mask_path)

with torch.no_grad():
    a = model(torch.from_numpy(img).unsqueeze(0).type(torch.cuda.FloatTensor)/255)

x,y,_ = plt.hist(a["out"].data.cpu().numpy().flatten())

#print(x.argmax(axis=0))
threshold = y[x.argmax(axis=0)]
plt.hist(a['out'].data.cpu().numpy().flatten())
plt.show()

img = tifffile.imread(image_path)
img = img[:,:,:3]

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
    plt.imshow(prediction['out'].cpu().detach().numpy()[0][0]>threshold)
    plt.title('Segmentation Output')
    plt.axis('off')
    plt.show()


plot_figure(img, mask, a, threshold)