import pickle
import numpy as np
from keras.utils import to_categorical
from lib.Cart2Pixel import Cart2Pixel
from lib.ConvPixel import ConvPixel


XGlobal = []
YGlobal = []

XTestGlobal = []
YTestGlobal = []

def train_norm(param, dataset, norm):
    np.random.seed(param["seed"])
    print("modelling dataset")
    global YGlobal
    YGlobal = to_categorical(dataset["Classification"])
    del dataset["Classification"]
    global YTestGlobal
    YTestGlobal = to_categorical(dataset["Ytest"])
    del dataset["Ytest"]

    global XGlobal
    global XTestGlobal

    if not param["LoadFromJson"]:
        # norm
        Out = {}
        if norm:
            print('NORM Min-Max')
            print("done")

            dataset['Xtest'].values[dataset['Xtest'] > 1] = 1
            dataset['Xtest'].values[dataset['Xtest'] < 0] = 0

            print(dataset["Xtest"].max().max())
            print(dataset["Xtest"].min().min())

        # TODO implement norm 2
        print("trasposing")

        q = {"data": np.array(dataset["Xtrain"].values).transpose(), "method": param["Metod"],
             "max_A_size": param["Max_A_Size"], "max_B_size": param["Max_B_Size"], "y": np.argmax(YGlobal, axis=1)}
        print(q["method"])
        print(q["max_A_size"])
        print(q["max_B_size"])

        # generate images
        XGlobal, image_model, toDelete = Cart2Pixel(q, q["max_A_size"], q["max_B_size"], param["Dynamic_Size"],
                                                    mutual_info=param["mutual_info"], params=param, only_model=False)

        del q["data"]
        print("Train Images done!")
        # generate test set image
        if param["mutual_info"]:
            dataset["Xtest"] = dataset["Xtest"].drop(dataset["Xtest"].columns[toDelete], axis=1)

        dataset["Xtest"] = np.array(dataset["Xtest"]).transpose()
        print("generating Test Images")
        print(dataset["Xtest"].shape)

        XTestGlobal = [ConvPixel(dataset["Xtest"][:, i], np.array(image_model["xp"]), np.array(image_model["yp"]),
                                 image_model["A"], image_model["B"])
                       for i in range(0, dataset["Xtest"].shape[1])]

        print("Test Images done!")

    #     # saving testingset
    #     name = "_" + str(int(q["max_A_size"])) + "x" + str(int(q["max_B_size"]))
    #     if param["No_0_MI"]:
    #         name = name + "_No_0_MI"
    #     if param["mutual_info"]:
    #         name = name + "_MI"
    #     else:
    #         name = name + "_Mean"
    #     if image_model["custom_cut"] is not None:
    #         name = name + "_Cut" + str(image_model["custom_cut"])
    #     filename = param["dir"] + "test" + name + ".pickle"
    #     f_myfile = open(filename, 'wb')
    #     pickle.dump(XTestGlobal, f_myfile)
    #     f_myfile.close()
    # else:
    #     XGlobal = dataset["Xtrain"]
    #     XTestGlobal = dataset["Xtest"]
        train_filename = param["dir"] + param['dir'].split('/')[-2] + "_train.pickle"
        test_filename = param["dir"] + param['dir'].split('/')[-2] + "_test.pickle"

        # Save training set
        with open(train_filename, 'wb') as f_myfile:
            pickle.dump(XGlobal, f_myfile)
        
        # Save test set
        with open(test_filename, 'wb') as f_myfile:
            pickle.dump(XTestGlobal, f_myfile)

    else:
        XGlobal = dataset["Xtrain"]
        XTestGlobal = dataset["Xtest"]
    # GAN
    del dataset["Xtrain"]
    del dataset["Xtest"]
    XTestGlobal = np.array(XTestGlobal)
    image_size1, image_size2 = XTestGlobal[0].shape
    XTestGlobal = np.reshape(XTestGlobal, [-1, image_size1, image_size2, 1])
    YTestGlobal = np.argmax(YTestGlobal, axis=1)

    print("done")
    return 1
