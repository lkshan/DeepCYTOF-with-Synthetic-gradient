# DeepCyTOF

Repository for the paper "Gating mass cytometry data by deep learning" by Huamin Li, Uri Shaham, Kelly P. Stanton, Yi Yao, Ruth Montgomery, and Yuval Kluger.

The script FLOWCAP_CellClassifier.py and DeepCyTOF.py are the main demo scripts, and can be generally used for classification experiments. It was used to train 
all neural networks like feed-forward cell classifier, denoising autoencoder, and MMD-ResNets used for the CyTOF experiments reported in our manuscript.

We only provide two datasets here: MultiCenter_16samples and GvHD from FlowCAP-I in Data. To obtain the whole datasets, i.e., three CyTOF datasets and five datasets from FlowCAP-I, please contact Huamin Li at huamin.li@yale.edu.

The models used to produce the results in the manuscript for MultiCenter_16samples are saved in savemodels.

All scripts are written in Keras.

Any questions should be referred to Huamin Li, huamin.li@yale.edu.
