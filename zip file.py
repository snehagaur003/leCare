from zipfile import ZipFile
import os

# zip = "new-plant-diseases-dataset.zip"
# with ZipFile(zip, "r") as unzip:
#     unzip.extractall("dataset")
# print("Dataset Extracted !!")

f = "dataset"
count = 0
for root, dirs, files in os.walk(f):
    count += len(files)

print("Total files:", count)
