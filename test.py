from tifffile import imread, imwrite

x = imread("TestExperiment_Data\Images\dop10rgbih_32344_5657_1_nw_tile2048_2048.tif")
print(x.shape)

x = x[:,:,4:]

print(x.shape)
imwrite("C:/Users/Dinga/Desktop/test_fifth_channel.tif", x)