#!/bin/sh

# IMPORTANT: CHANGE sparm->num_features in svm_struct_latent_api.c
# (function_name: init_struct_model, Line 250) MANUALLY OTHERWISE YOU WILL GET WRONG MODEL

tot_rlabs=1
current_lab=100

# Copy randomlabel_x.txt from the path given below
source_dir='/home/ssabidi/Shaukat/WellSSVM/experiments/HMM_Label_generation/ozone_dataset/Dim_3/onehour'

while [[ $current_lab -ge $tot_rlabs ]];do

#clear /params folder
rm -f /home/ssabidi/Shaukat/WellSSVM/experiments/latentssvm_HCRF/model.txt
rm -f /home/ssabidi/Shaukat/WellSSVM/experiments/latentssvm_HCRF/results.txt

# Run LatentSSVM learning and classification module
cd /home/ssabidi/Shaukat/WellSSVM/experiments/latentssvm_HCRF
make clean
make
./svm_latent_learn -e 0.1 -c 10 $source_dir/text_files/randomlabel_$current_lab.txt /home/ssabidi/Shaukat/WellSSVM/experiments/latentssvm_HCRF/model.txt > $source_dir/text_files/predictions/latent_SSVM_C10/learn_rlab_$current_lab.txt
./svm_latent_classify $source_dir/groundtruth_file.txt /home/ssabidi/Shaukat/WellSSVM/experiments/latentssvm_HCRF/model.txt results.txt > $source_dir/text_files/predictions/latent_SSVM_C10/$current_lab.txt

# Copying model file to storage location
cp /home/ssabidi/Shaukat/WellSSVM/experiments/latentssvm_HCRF/model.txt $source_dir/text_files/predictions/latent_SSVM_C10/

# Renaming model file
mv $source_dir/text_files/predictions/latent_SSVM_C10/model.txt $source_dir/text_files/predictions/latent_SSVM_C10/model_$current_lab.txt


((current_lab--))

done
