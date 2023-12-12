import torch
import matplotlib.pyplot as plt
import tifffile
import pandas as pd

model_MSE = torch.load("./Training_5ch/weights.pt")
model_BCE = torch.load("./Training_5ch/weights.pt")
model_MSE.eval()
model_BCE.eval()

df_MSE = pd.read_csv("./Training_5ch/log.csv")
df_BCE = pd.read_csv("./Training_5ch_BCE/log.csv")

# df_MSE.plot(x="epoch", figsize=(15,8))
# plt.show()

# df_BCE.plot(x='epoch',figsize=(15,8));
# plt.show()

# print(df_MSE[['Train_auroc','Test_auroc']].max())
# print(df_BCE[['Train_auroc','Test_auroc']].max())

#image_name = "dop10rgbih_32363_5651_1_nw_tile0_7168.tif"
#img = tifffile.imread(f"C:/Users/dinga/Documents/Doktorat/2023_CD_TrainData_cleansed/02_test_images/{image_name}").transpose(2,0,1)
#mask = tifffile.imread(f"C:/Users/dinga/Documents/Doktorat/2023_CD_TrainData_cleansed/02_test_masks/{image_name}")

image_path = "C:/Users/dinga/Documents/Doktorat/DeepLabV3/TrainingData/Images/dop10rgbih_32363_5651_1_nw_tile0_7168.tif"
mask_path = "C:/Users/dinga/Documents/Doktorat/DeepLabV3/TrainingData/Masks/dop10rgbih_32363_5651_1_nw_tile0_7168.tif"
img = tifffile.imread(image_path).transpose(2,0,1)
mask = tifffile.imread(mask_path)

with torch.no_grad():
    a = model_BCE(torch.from_numpy(img).unsqueeze(0).type(torch.cuda.FloatTensor)/255)
    b = model_MSE(torch.from_numpy(img).unsqueeze(0).type(torch.cuda.FloatTensor)/255)

plt.hist(a['out'].data.cpu().numpy().flatten())
plt.show()

plt.hist(b['out'].data.cpu().numpy().flatten())
plt.show()

#img = tifffile.imread(f"C:/Users/dinga/Documents/Doktorat/2023_CD_TrainData_cleansed/02_test_images/{image_name}")
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


plot_figure(img, mask, a, 0)
plot_figure(img, mask, b, 0.5)