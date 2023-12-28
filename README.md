# Chemical Structure Elucidation (CSE) from Mass Spectrometry and Chemical Formula
## Abstract: 
 Chemical structure elucidation (CSE) is the process of deducing the structural formula of an unknown chemical compound, with mass spectrometry (MS) and Nuclear Magnetic Resonance (NMR) being the two most widely used techniques. By identifying the fragments produced when the unknown chemical compound breaks up inside the mass spectrometer, and with the complementary data provided by NMR, chemists can infer the structure of the molecule. However, manually carrying out this process is known to be a tedious task in the field of analytical chemistry. This project aims to use machine learning to aid chemists in CSE by predicting the structure of the model given MS, NMR or any other measurable signals.

## Reproducibility Notes 
### Additional setup 
 There was a 'canopus' folder containing training data from CANOPUS, not commited to GitHub due to its large file size. It can be downloaded at this link: [https://drive.google.com/file/d/1e7dVIjO5AaG84Ct3xbwv3tzyWeMvU7Y8/view?ts=64c362e7] 
 In addition to those, there are fragmentation trees to some samples, to be extracted and placed into the canopus folder under a subfolder "treeviews". It is downloadable at this link: [https://drive.google.com/file/d/1Z5nLH4RCyWBITao4-VR2oct_5oDVzska/view?usp=drive_link] 
 Make sure that pytorch and dgl are both installed with CUDA, and that the other python libraries specified in requirements.txt are downloaded. 
 Run file_structure.py under the folder RL_attempt to create the necessary directories. 

### Training the FTree GCN 
 To train the FTree GCN, run 
