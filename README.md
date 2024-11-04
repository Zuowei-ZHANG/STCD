# STCD
This is the available code for the paper ["Style Transfer-based Unsupervised Change Detection from Heterogeneous Images"].

# Requirements
PyTorch==2.1.0+cu126, python==3.9.18

# To test the STCD model
The file main.py obtained the ultimate results, which contains parameters for selecting the dataset and differentbenchmark models.

# To generate the confusing maps
The CDTool.py generates the confusing maps based on change maps and Ground Truth.

# Data set description
We use four datasets to demonstrate the effectiveness of STCD. Due to the different heterogeneity of image pairs, Gloucester and Shuguang Village are multi-source datasets obtained from different types of sensors. Sardinia and Texas
are multisensor datasets acquired by different optical sensors. Besides, all datasets contain a pre-event image, a post-event image, and the corresponding reference image. TABLE I in the paper describes these datasets in detail.

For convenience，we have collected these datasets in Baidu Cloud with Link (https://pan.baidu.com/s/1rK3IMl6Ml9kT4oO9oMLxCA?pwd=xw3r). The extraction code is: xw3r.
