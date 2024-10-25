# Exploring a novel panel of multi-omic blood biomarkers for preclinical Alzheimer’s disease with the Australian Imaging, Biomarker and Lifestyle (AIBL) cohort

This is the official repository accompanying the paper **Exploring a novel panel of multi-omic blood biomarkers for preclinical Alzheimer’s disease with the Australian Imaging, Biomarker and Lifestyle (AIBL) cohort**

**Authors:**

Yihan Wang, Alanna G. Spiteri, Katherine Jeppe, Joel R. Steele, Christopher K. Barlow, Ralf Schittenhelm, Colin L. Masters, Benjamin Goudey, Liang Jin, Yijun Pan, and the AIBL Research Group


## Overview
We utilized data from 90 participants in the AIBL cohort, all of whom had completed the ORCA (Online Repeatable Cognitive Assessment) learning test. The study incorporated *proteomics* data consisting of 596 high/mid-confidence features, as well as *metabolomics* data with 1,345 peak intensity features. Additionally, *demographic* factors such as age, gender, education, and APOE4 status (yes/no) were included in the analysis. The primary aims of the study were: 1) to use minimally invasive data to predict amyloid-beta status (high vs. normal) based on PET imaging; and 2) to identify potential biomarkers that can distinguish participants in the preclinical phase of Alzheimer's disease.


## General pipeline
![general pipeline](./plots/general_pipeline.png)

## Overall Data Description
![basic statistics](./plots/basic-statistics.png)

## ML construction pipeline
![ML construction pipeline](./plots/ML_construction.png)

## Acknowledgements
The study received support from National Health and Medical Research Council, Australia (GNT2007912), Alzheimer’s Association, USA (23AARF-1020292), and Bethlehem Griffiths Research Foundation (BGRF2306).

## Data Sharing Statement
The data used in this study are derived from the Australian Imaging, Biomarkers, and Lifestyle (AIBL) dataset, which is subject to specific access conditions. Due to these restrictions, we are unable to share the data directly.

## Code Repository Structure
- [ML_code](/ML_code): Contains the code for machine learning model construction and evaluation.
    - [preprocessing](/ML_code/preprocessing.ipynb): Scripts for merging proteomics and metabolomics data, as well as different versions of dataset construction (e.g., using omics data alone, combined with demographic data, and imputation applied exclusively to demographic information for handling missing values)
    - [plots-R](/ML_code/plots.Rmd): R code for generating basic statistical plots (e.g., population pyramid).
    - [plots-python](/ML_code/plots.ipynb): Python code for generating basic statistical summaries and visualizations (e.g., PCA elbow plots, volcano plots, heatmaps, and violin plots for top features).
    - [model-evaluation](/ML_code/Model-evaluation-IQR.ipynb): Code for machine learning model construction and evaluation, focused on logistic regression (LR) models.
    - [model-evaluation-full](/ML_code/Model-evaluation-IQR-full.ipynb): Complete machine learning model construction and evaluation for all models considered in the study.
- [results](/results): Saved results from the machine learning models, corresponding to the supplementary document on model performance.
- [plots](/plots): Repository of various plots generated during model evaluation and analysis.



## Environment required
- python environment: 3.11
- statannotations: 0.5.0
- Seaborn: 0.11.2
- sklearn: 1.3.1
- Feyn (Qlattice): 3.2.0
    - Abzu document: https://docs.abzu.ai/ 
    - QLattice: use `pip3 install feyn`
- mRMR: 
    - mRMR document: https://github.com/smazzanti/mrmr
    - install: `pip install mrmr_selection`


