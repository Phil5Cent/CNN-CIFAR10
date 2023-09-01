"read each array and divide all by 255, "

#AT THE END WE ARE HOPING FOR THE SHAPE TO BE 50000, 3072, HENCE CREATING A GIANT 50000 IMAGE DATASET

import pickle
import numpy as np
import os

parent_path = os.path.dirname(os.path.abspath(__file__))


for i in range(5):
    with open(f"{parent_path}/data_batch_{i+1}", "rb") as file:
        dataset = pickle.load(file, encoding="bytes")
        
        
        Label = dataset[b"labels"]
        Data = dataset[b"data"]
        
        #print(len(Data))
        #Data = Data.reshape(10000,1024,3)
        #Data.transpose((0,2,1))
        
        




        red = Data[:, :1024].reshape(-1, 32, 32, 1)
        green = Data[:, 1024:2048].reshape(-1, 32, 32, 1)
        blue = Data[:, 2048:].reshape(-1, 32, 32, 1)

# Concatenate the red, green, and blue channels along the last axis
        Data = np.concatenate([red, green, blue], axis=-1)









        #for i in range(len(Data)):
        #    Data[i].reshape(3,1024)
        #    Data[i].transpose()

        try:
            TData = np.concatenate((TData,Data))
            TLabel = np.concatenate((TLabel,Label))
        except:
            TData = Data
            TLabel = Label
        "TData = np.append(TData,Data,axis=2)"
       
        #print(f"Opened data set {i+1} without issue!")
        #

TData = np.divide(TData,255)

LabelArray = np.zeros((len(TLabel), 10))

#print(np.shape(LabelArray))

for i in range(len(TLabel)):
    LabelArray[i,TLabel[i]] = 1




#print(TLabel[0:50])

#print(LabelArray[0:5])
#print(len(TLabel))
#print(np.shape(TLabel))










with open("TrainingData.npy", "wb") as file:
    np.save(file,TData)


with open("TrainingLabels.npy", "wb") as file:
    np.save(file, LabelArray)







#TrainingData = [TLabel.tolist(),TData.tolist()] #MAKE THIS A DICTIONARY INSTEAD OF A LIST, LIST CANT PRESERVE SHAPE, AND NP CANT HANDLE MISMATCH DIMENSIONS. OR FIND SOME OTHER WAY
#TO SAVE THE INFORMATION THAT YOU WILL LATER BE OPENING

#print(type(TrainingData))



#rint(TData)

#print (type((dataset[b"data"])[0]))


#print(((dataset[b"data"])[0]).size)

#with open("TrainingData", "w") as f:


 #   f.write(np.array2string((dataset[b"data"])[0]))



