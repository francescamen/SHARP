# SHARP
Algorithms for human activity recognition with a commercial IEEE 802.11ac router @ 5 GHz, 80 MHz of bandwidth.

This repository contains the reference code for the article [''SHARP: Environment and Person Independent Activity Recognition with Commodity IEEE 802.11 Access Points''](https://arxiv.org/abs/2103.09924).

If you find the project useful and you use this code, please cite our article:
```
@article{meneghello2022sharp,
  author = {Meneghello, Francesca and Garlisi, Domenico and Dal Fabbro, Nicolò and Tinnirello, Ilenia and Rossi, Michele},
  title = {{SHARP: Environment and Person Independent Activity Recognition with Commodity IEEE 802.11 Access Points}},
  year = {2022}
}
```

## How to use
Clone the repository and enter the folder with the python code:
```bash
cd <your_path>
git clone https://github.com/signetlabdei/SHARP
```

Download the input data from http://researchdata.cab.unipd.it/id/eprint/624 and unzip the file.
For your convenience, you can use the ```input_files``` inside this project folder to place the files but the scripts work whatever is the source folder.

The dataset contains the Wi-Fi channel frequency response (CFR) data collected in an IEEE 802.11ac network through [NEXMON CSI](https://github.com/seemoo-lab/nexmon_csi). 
The information is collected by a monitor node (ASUS RT-AC86U router) while two terminals are exchanging traffic in channel 42 (5.21 GHz for the center frequency and 80 MHz of bandwidth) and a person acts as an obstacle for the transmission by performing different activities. 
The considered movements are the following: walking (W) or running (R) around, jumping (J) in place, sitting (L) or standing (S) somewhere in the room, sitting down and standing up (C) continuously, and doing arm gym (H).
The CFR data for the empty room (E) is also provided. We obtained data from three volunteers, a male, and two females.
The complete description of the dataset can be found in the reference paper.

The code for SHARP is implemented in Python and can be found in the ```Python_code``` folder inside this repository. The scripts to perform the processing are described in the following, together with the specific parameters.

### Phase sanitization
The following three scripts encode the phase sanitization algorithm detailed in Section 3.1 of the referred article.
```bash
python CSI_phase_sanitization_signal_preprocessing.py <'directory of the input data'> <'process all the files in subdirectories (1) or not (0)'> <'name of the file to process (only if 0 in the previous field)'> <'number of spatial streams'> <'number of cores'> <'index where to start the processing for each stream'> 
```
e.g., python CSI_phase_sanitization_signal_preprocessing.py ../input_files/S1a/ 1 - 1 4 0

```bash
python CSI_phase_sanitization_H_estimation.py <'directory of the input data'> <'process all the files in subdirectories (1) or not (0)'> <'name of the file to process (only if 0 in the previous field)'> <'number of spatial streams'> <'number of cores'> <'index where to start the processing for each stream'> <'index where to stop the processing for each stream'> 
```
e.g., python CSI_phase_sanitization_H_estimation.py ../input_files/S1a/ 0 S1a_E 1 4 0 -1

```bash
python CSI_phase_sanitization_signal_reconstruction.py <'directory of the processed data'> <'directory to save the reconstructed data'> <'number of spatial streams'> <'number of cores'> <'index where to start the processing for each stream'> <'index where to stop the processing for each stream'> 
```
e.g., python CSI_phase_sanitization_signal_reconstruction.py ./phase_processing/ ./processed_phase/ 1 4 0 -1

### Doppler computation
The following script computes the Doppler spectrum as described in Section 3.2 of the referred article.

```bash
python CSI_doppler_computation.py <'directory of the reconstructed data'> <'sub-directories of data'> <'directory to save the Doppler data'> <'starting index to process data'> <'end index to process data (samples from the end)'> <'number of packets in a sample'> <'number of packets for sliding operations'> <'noise level'> <--bandwidth 'bandwidth'>
```
e.g., python CSI_doppler_computation.py ./processed_phase/ S1a ./doppler_traces/ 800 800 31 1 -1.2

### Dataset creation
- Create the datasets for training and validation
```bash
python CSI_doppler_create_dataset_train.py <'directory of the Doppler data'> <'sub-directories, comma-separated'> <'number of packets in a sample'> <'number of packets for sliding operations'> <'number of samples per window'> <'number of samples for window sliding'> <'labels of the activities to be considered'> <'number of streams * number of antennas'>
```
  e.g., python CSI_doppler_create_dataset_train.py ./doppler_traces/ S1a 100 1 340 30 E,L,W,R,J,C 4

- Create the datasets for test
```bash
python CSI_doppler_create_dataset_test.py <'directory of the Doppler data'> <'sub-directories, comma-separated'> <'number of packets in a sample'> <'number of packets for sliding operations'> <'number of samples per window'> <'number of samples for window sliding'> <'labels of the activities to be considered'> <'number of streams * number of antennas'>
```
  e.g., python CSI_doppler_create_dataset_test.py ./doppler_traces/ S7a 100 1 340 30 E,L,W,R,J,C 4

### Train the learning algorithm for HAR
```bash
python CSI_network.py <'directory of the datasets'> <'sub-directories, comma-separated'> <'length along the feature dimension (height)'> <'length along the time dimension (width)'> <'number of channels'> <'number of samples in a batch'> <'name prefix for the files'> <'activities to be considered, comma-separated'> <--bandwidth 'bandwidth'> <--sub-band 'index of the sub-band to consider (for 20 MHz and 40 MHz)'> 
```
e.g., python CSI_network.py ./doppler_traces/ S1a 100 340 1 32 4 network E,L,W,R,J,C

### Use the trained algorithm for inference
- Run the algorithm with the test data 
```bash
python CSI_network_test.py <'directory of the datasets'> <'sub-directories, comma-separated'> <'length along the feature dimension (height)'> <'length along the time dimension (width)'> <'number of channels'> <'number of samples in a batch'> <'name prefix for the files'> <'activities to be considered, comma-separated'> <--bandwidth 'bandwidth'> <--sub-band 'index of the sub-band to consider (for 20 MHz and 40 MHz)'> 
```
  e.g., python CSI_network_test.py ./doppler_traces/ S7a 100 340 1 32 4 network E,L,W,R,J,C

- Compute the performance metrics using the output file of the test
```bash
python CSI_network_metrics.py <'name of the output file containing the metrics'> <'activities to be considered, comma-separated'>
```
  e.g., python CSI_network_metrics.py complete_different_E,L,W,R,J,C_S7a_band_80_subband_1 E,L,W,R,J,C 

- Plot the performance metrics
```bash
python CSI_network_metrics_plots.py <'sub-directories, comma-separated'>
```
  e.g., python CSI_network_metrics_plots.py complete_different_E,L,W,R,J,C_S7a_band_80_subband_1 E,L,W,R,J,C

### Parameters
The results of the article are obtained with the parameters reported in the examples. For convenience, the repository also contains two pre-trained networks, i.e., ``single_ant_network.h5`` and ``single_ant_smart_home_network.h5`` respectively for 5-classes and 8-classes classification problems.

### Python and relevant libraries version
Python >= 3.7.7  
TensorFlow >= 2.6.0  
Numpy >= 1.19.5  
Scipy = 1.4.1  
Scikit-learn = 0.23.2  
OSQP >= 0.6.1

## Contact
Francesca Meneghello
meneghello@dei.unipd.it
github.com/francescamen
