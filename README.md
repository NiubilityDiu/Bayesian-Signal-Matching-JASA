# JASA-A&CS Reproducibility Materials for "Bayesian Signal Matching  for Transfer Learning in ERP-Based Brain Computer Interface"

## Author
Blinded during peering review.

## Overview

This github repository contains the code and relevant outputs to reproduce key findings presented in the paper "Bayesian Signal Matching for Transfer Learning in ERP-Based Brain Computer Interface." The reproduction focuses on the real-data based simulation studies and the real data analysis, including Figure 3, Figure 5, Table 2 in the main text and Figure S7 in the supplementary material.

## Workflow
* Download the entire repository to your local desktop and rename "BSM-Code-V2."
* Go to Python folder and open the following two jupyter notebooks:
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
  * The data dictionary (as well as experimental design) can be found in this paper [“A plug-and-play brain-computer interface to operate commercial assistive technology”](https://doi.org/10.3109/17483107.2013.785036).
 
  * K151 is the single participant we focus on in Section 5. Under `K151`, there are five subfolders:
    * `mixture_gibbs_letter_5_reduced_xdawn` (BSM-Mixture),
    * `borrow_gibbs_letter_5_reduced_xdawn` (BSM),
    * `reference_numpyro_letter_5_xdawn` (BSM-Reference),
    * `MDWM` (MDWM),
    * `swLDA` (swLDA).
  * Under `FRT_files`, there are two subfolders, `K151` and `Prediction`. The former one stores session-specific (`001_BCI_FRT`, `002_BCI_FRT`, `003_BCI_FRT`) prediction accuracy using each method, while the latter one stores the prediction accuracy combining sessions of all eight participants with neuromuscular diseases and brain injuries (shown in Figure 6).
* For simulation studies, we provide minimally replicable amount of information for naive single-, multi-channel, and real-data-based scenarios under (`N_7_K_3/N_7_K_3_option_1_sigma_3.0_rho_0.6`, `N_7_K_3/N_7_K_3_option_2_sigma_3.0_rho_0.6`), (`N_7_K_3/N_7_K_3_multi_option_1`, `N_7_K_3/N_7_K_3_multi_option_2`), and `N_24_K_24_multi_xdawn_eeg`, respectively.
  * For single- and multi-channel naive simulation scenarios, only aggregated `inference_summary` and `prediction_summary` are stored.
  * For real-data-based multi-channel scenario, a particular replication `iter_0` with sample datasets and `prediction_summary` across 100 replications are stored.

## Software Environment
* Python Version is 3.7 on local desktop and 3.11 on the institutional server.
  * PyCharm 2024.02.01 is used to locally run Python codes. If you have an educational email address, you can download it for free [here](https://www.jetbrains.com/pycharm/).
   * It seems that Pycharm Education does not support jupyter notebook for free. Please establish a virtual environment under the `Python` folder. 
  * Relevant packages include `numpy (1.21.6)`, `scipy (1.7.3)`, `seaborn (0.12.1)`, `matplotlib (3.5.3)`, `os`, `tqdm (4.64.1)`, `jax (0.3.25)`, `jaxlib`, `numpyro (0.10.1)`, `scikit-learn (1.0.2)`, `pyriemann (0.5)`, `mne (1.3.1)`, `notebook`, and `json (4.17.3)`.
* MATLAB Version R2022b or newer. No particular toolboxes are required to download.
* R Version 4.2.2 (2022-10-31) with platform: x86_64-apple-darwin17.0 (64-bit). Relevant R packages using `sessionInfo()` include
  * attached base packages: `stats`, `graphics`, `grDevices`, `utils`, `datasets`, `methods`, and `base`.
  * other attached packages: `gridExtra_2.3`, `ggplot2_3.5.1`, and `R.matlab_3.7.0`.
  * loaded via a namespace (and not attached): `fansi_1.0.3`, `ithr_2.5.0`, `dplyr_1.1.4`, `utf8_1.2.2`       
 `R.methodsS3_1.8.2`, `grid_4.2.2`, `R6_2.5.1`, `lifecycle_1.0.3`, `gtable_0.3.1`, `magrittr_2.0.3`,
 `scales_1.3.0`, `pillar_1.9.0`, `rlang_1.1.3`, `cli_3.6.2`, `R.utils_2.12.2`, `R.oo_1.25.0`,
 `generics_0.1.3`, `vctrs_0.6.5`, `tools_4.2.2`, `glue_1.6.2`, `munsell_0.5.0`, `compiler_4.2.2`,
 `pkgconfig_2.0.3`, `colorspace_2.0-3`, `tidyselect_1.2.0`, and `tibble_3.2.1`.  

