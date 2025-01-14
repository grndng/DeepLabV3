# change for training
# possible layer stacks:
# "dsm", "rgb", "rgbh", "rgbi", "rgbih"
layer_stack = "rgbih"

match layer_stack:
    case "dsm":
        channels = 1
    case "rgb":
        channels = 3
    case "rgbih":
        channels = 5
    case other:
        channels = 4

exp_directory = "D:/Potsdam_Final/Training_Results/Potsdam_512_RGBIH"
