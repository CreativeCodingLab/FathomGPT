import os
import json

# directory name from which
# we are going to extract our files with its size
path = "data/fathomnet"
 
# Get list of all files only in the given directory
fun = lambda x : os.path.isfile(os.path.join(path,x))
files_list = filter(fun, os.listdir(path))
 
# Create a list of files in directory along with the size
size_of_file = [
    (f,os.stat(os.path.join(path, f)).st_size)
    for f in files_list
]

sizes = {}
for f, s in size_of_file:
    sizes[f.replace('.json', '')] = s

with open("concept_sizes.json", "w") as outfile:
    json.dump(sizes, outfile)

