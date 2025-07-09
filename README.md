# JASA-A&CS Reproducibility Materials for "Bayesian Signal Matching  for Transfer Learning in ERP-Based Brain Computer Interface"

## Author
Blinded during peering review

## Overview

This github repository contains the code and relevant outputs to reproduce key findings presented in the paper "Bayesian Signal Matching for Transfer Learning in ERP-Based Brain Computer Interface." The reproduction focuses on the real-data based simulation studies and the real data analysis, including Figure 3, Figure 5, Table 2 in the main text and Figure S7 in the supplementary material.

## Workflow

Go to Python folder and open the following two jupyter notebooks:
* [SIM_multi_24_demo.ipynb](https://github.com/NiubilityDiu/Bayesian-Signal-Matching-JASA/blob/main/Python/SIM_multi_24_demo.ipynb)
* [EEG_multi_24_demo.ipynb](https://github.com/NiubilityDiu/Bayesian-Signal-Matching-JASA/blob/main/Python/EEG_multi_24_demo.ipynb)

## Repository Structure
### Code
In this study, codes are written in Python, R, and MATLAB. They are saved under `Python`, `R`, and `MATLAB` folders, respectively. 
* MATLAB code is used to perform EEG signal pre-processing and implement the swLDA method.
  * `EEG_bandpass_filter.m` is used to implement the pre-processing shown in Section 5.1.
  * `EEG_train_swLDA_matlab_all.m` is used to implement the swLDA method for real EEG signals including all 16 channels in Section 5.
  * `SIM_cluster_swLDA_MATLAB_multi_24.m` is used to implement the swLDA method for real-data-based simulation studies in Section 4.2.
  * `SIM_train_swLDA_matlab.m` and `SIM_train_swLDA_matlab_multi.m` are used to implement the swLDA method for naive simulation studies with single and multiple channels in Section S4.1 and 4.1, respectively.
* R codes are primarily used to visualize the prediction and inference results.
  * `eeg_prediction_3way_frt_xdawn_multi.R` is used to summarize and visualize testing prediction accuracy of real data analysis in Section 5.
  * `eeg_prediction_frt_xdawn_multi_sensitivity.R` is used to compute testing prediction accuracy of K151 with nine combinations of hyper-parameters for senstivity analysis in Section S5.3.
  * `sim_summary_3way_multi_24.R` is used to summarize and visualize testing prediction accuracy of the real-data based simulation studies in Section 4.
* The BSM-related and MDWM algorithms are coded in Python.
  * Files starting with `EEG` refer to the real data analysis, while files starting with `SIM` refer to the simulation studies.
  * Files ending with nothing special refer to the naive single-channel simulation studies. Files ending with `_multi` refer to the naive multi-channel simulation studies. Finally, files ending with `_multi_24` refer to the real-data-based multi-channel simultion studies.
    * For example, `SIM_signal_parameter.py`, `SIM_signal_parameter_multi.py`, and `SIM_signal_parameter_multi_24.py` refer to the data generation codes for naive single-channel, naive multi-channel, and real-data-based multi-channel simulation studies, respectively.
  * Files starting with `numpyro` are BSM-Reference-based methods that are implemented using `numpyro` package automaticlly, while files starting with `gibbs` are BSM methods that are implemented with self-written MCMC functions.
* Two jupyter notebooks with detailed instructions are available for the readers. 

### Data
All relevant data are saved under `EEG_MATLAB_data` folder. This folder further contains three subfolders: `SIM_files` (simulation studies), `TRN_files` (training sets of real data analysis), and `FRT_files` (free-typing or testing sets of real data analysis).
* For real data analysis, since the original participants in the study did not consent to release their data in public directly, even for the de-identified version, those files are only available upon request for the academic audiences. Therefore, no raw EEG datasets are available.
  * K151 is the single participant we focus on in Section 5. Under `K151`, there are five subfolders:
    * `mixture_gibbs_letter_5_reduced_xdawn` (BSM-Mixture),
    * `borrow_gibbs_letter_5_reduced_xdawn` (BSM),
    * `reference_numpyro_letter_5_xdawn` (BSM-Reference),
    * `MDWM` (MDWM),
    * `swLDA` (swLDA).
  * Under `FRT_files`, there are two subfolders, `K151` and `Prediction`. The former one stores session-specific (`001_BCI_FRT`, `002_BCI_FRT`, `003_BCI_FRT`) prediction accuracy using each method, while the latter one stores the prediction accuracy combining sessions of all eight participants with neuromuscular diseases and brain injuries (shown in Figure 6).
* For simulation studies, we provide part of information for naive single-, multi-channel, and real-data-based scenarios under (`N_7_K_3/N_7_K_3_option_1_sigma_3.0_rho_0.6`, `N_7_K_3/N_7_K_3_option_2_sigma_3.0_rho_0.6`), (`N_7_K_3/N_7_K_3_multi_option_1`, `N_7_K_3/N_7_K_3_multi_option_2`), and `N_24_K_24_multi_xdawn_eeg`, respectively. For single- and multi-channel naive simulation scenarios, only aggregated `inference_summary` and `prediction_summary` are stored. For real-data-based multi-channel scenario, a `iter_0` and `prediction_summary` across 100 replications are stored.



