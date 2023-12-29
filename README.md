# Chemical Structure Elucidation (CSE) from Mass Spectrometry and Chemical Formula

## Abstract: 

Chemical structure elucidation (CSE) is the process of deducing the structural formula of an unknown chemical compound, with mass spectrometry (MS) and Nuclear Magnetic Resonance (NMR) being the two most widely used techniques. By identifying the fragments produced when the unknown chemical compound breaks up inside the mass spectrometer, and with the complementary data provided by NMR, chemists can infer the structure of the molecule. However, manually carrying out this process is known to be a tedious task in the field of analytical chemistry. This project aims to use machine learning to aid chemists in CSE by predicting the structure of the model given MS, NMR or any other measurable signals.


## Reproducibility Notes 

### Additional setup 

There was a 'canopus' folder containing training data from CANOPUS, not commited to GitHub due to its large file size. It can be downloaded at this link: [https://drive.google.com/file/d/1e7dVIjO5AaG84Ct3xbwv3tzyWeMvU7Y8/view?ts=64c362e7] 
In addition to those, there are fragmentation trees to some samples, to be extracted and placed into the canopus folder under a subfolder "treeviews". It is downloadable at this link: [https://drive.google.com/file/d/1Z5nLH4RCyWBITao4-VR2oct_5oDVzska/view?usp=drive_link] 
Make sure that pytorch and dgl are both installed with CUDA, and that the other python libraries specified in requirements.txt are downloaded. 
Run file_structure.py under the folder RL_attempt to create the necessary directories. 


### Training the models 

To train the FTree GCN and main models, run train_mass_spec.py 
Then, run train_main_model.py to train the rest, selecting a FTree GCN model to use. (random seed reset to here) 


### Testing the FTree GCN 

To test the FTree GCN, run mass_spec_visualization.py 


### Plotting the main model's accuracy in detection 

To plot a histogram for the main model's accuracy in aomaly detection, to visualize its effectiveness in separating normal and anomalous states, run visualize_scores_anomaly.py 


### Testing the model's ability to generate molecules 

To test the main models' ability to search and generate the molecule from a starting state of only aromatic compounds, run test_full_search_main_models.py (alternatively, run the three similarly named programs test_full_search_main_models_max_12_filtered_0_1.py, test_full_search_main_models_max_12.py, and test_full_search_main_models_max_15.py), changing variables (e.g. the list epoch_nums, to choose which epochs of the main model to test) as desired. Then, run plot_full_search_main_models_results.py to see the visualization of the results. 

To test the main models' accuracy in searching and generating the molecule from different depths, run test_depth_astar_search.py (alternatively, run the three similarly named programs test_depth_astar_search_max_12_filtered_0_1.py, test_depth_astar_search_max_12.py, and test_depth_astar_search_max_15.py), changing variables as desired. Then, run plot_depth_main_models_results.py to see the visualization of the results. 

