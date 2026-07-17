import os

classes = sorted(os.listdir("./dataset_small/train"))

for i, cls in enumerate(classes):
    print(i, cls)