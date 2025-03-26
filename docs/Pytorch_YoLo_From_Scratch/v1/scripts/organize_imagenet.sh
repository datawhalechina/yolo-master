#!/bin/bash

# Check that an argument was provided. Otherwise, print the usage of the program and terminate the script.
if (( $# != 1 )); then
  echo "Usage: ./organize_imagenet.sh <dir_path>"
  echo "Extract the ImageNet data from the train and the validation .tar files to a train/ and a val/ directory, " \
       "respectively. The ILSVRC2012_img_train.tar contains one .tar file for each of the 1000 classes of the ImageNet"\
       " dataset. Each of the .tar class files contains the JPEG images of the class. The ILSVRC2012_img_val.tar " \
       "contains The validation images. The images must be moved into appropriate subfolders."

  echo
  echo "Arguments:"
  echo -e "\t<dir_path>: The path of the directory that contains the downloaded .tar files (ILSVRC2012_img_train.tar,"\
          " ILSVRC2012_img_val.tar)"
  exit 1
fi

# Create a new train/ directory and extract the train .tar file in this directory
TRAIN_DIR=$(realpath -m "$1"/train)
mkdir -p "$TRAIN_DIR"
tar -xvf $(realpath -m "$1"/ILSVRC2012_img_train.tar) -C "$TRAIN_DIR" > /dev/null

# For each of the <class_name>.tar files in the train/ directory:
# create a new sub-directory <class_name>/
# extract the <class_name>.tar file in this directory
# delete the <class_name>.tar file
for f in $(realpath -m "$TRAIN_DIR"/*); do
  CLASS_NAME=$(echo "$f" | sed 's|.*/||; s|\.tar||')
  CLASS_DIR=$(realpath -m "$TRAIN_DIR/$CLASS_NAME")
  mkdir -p "$CLASS_DIR"
  tar -xvf "$f" -C "$CLASS_DIR" > /dev/null
  rm -rf "$f"
done

# Create a new val/ directory and extract the val .tar file in this directory
VAL_DIR=$(realpath -m "$1"/val)
mkdir -p "$VAL_DIR"
tar -xvf $(realpath -m "$1"/ILSVRC2012_img_val.tar) -C "$VAL_DIR" > /dev/null

# Create the subfolders of the val/ directory and organize the images. To achieve this, we will run the valprep.sh
# script from https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
CURR_DIR=$(pwd)
cd "$VAL_DIR"
wget -qO - https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd "$CURR_DIR"
