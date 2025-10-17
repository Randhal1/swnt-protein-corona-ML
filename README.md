# swnt-protein-corona-ML
Ensemble classifiers and data scraping tools for the prediction of protein surface adsorption to single-walled carbon nanotubes

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5640140.svg)](https://doi.org/10.5281/zenodo.5640140)   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5641450.svg)](https://doi.org/10.5281/zenodo.5641450)


Codebase to pair with "Supervised Learning Model Predicts Protein Adsorption to Carbon Nanotubes" by Nicholas Ouassil*, Rebecca L. Pinals*, Jackson Travis Del Bonis-O'Donnell, Jeffrey W. Wang, and Markita P. Landry in [*biorXiv*](https://doi.org/10.1101/2021.06.19.449132) 

*Co-Authors 


## Included Files and What They Do

1. **01-prep_for_netsurfp.ipynb** (Jupyter Notebook) 

    * This script collects data for processing with **NetSurfP 2.0** it will generate a text file. Copy this text file into the prompt on the NetSurfP 2.0 website. Download data and place into an excel sheet then revisit this script and let it process the excel sheet. 

2. **02-Power_Study_Script.ipynb** (Jupyter Notebook)

   * Reproduces the labeling scheme used in figure 1b of the published work.
   * *Requires* the output of __01-prep_for_netsurfp.ipynb__ 

3. **03-data_workup_script.ipynb** (Jupyter Notebook)
    
   * This script is used to do the majority of the prepping for running through 
   * *Requires* the output of __01-prep_for_netsurfp.ipynb__ and Power Value from __02-Power_Study_Script.ipynb__

4. **04-classification_script.ipynb** (Jupyter Notebook)

    * Reproduces the classification experiments from the manuscript
    * *Requires* the output of __03-data_workup_script.ipynb__ 


5. **041-hyperparameter_optimization.ipynb** (Jupyter Notebook)

    * Uses GridSearchCV to find the best hyperparameters. Be Careful here as runtimes can rapidly increase with too many parameters
    * *Requires* the output of __03-data_workup_script.ipynb__ 

6. **042-different_classifier_testing.ipynb** (Jupyter Notebook)

    * Uses Generic Architectures to Identify the Best Architecture for the Data
    * *Requires* the output of __03-data_workup_script.ipynb__ 

6. **Data Prep Functions, Interpro Scraping, UniProt NetSurfP Scraping** (Python Scripts)

    * Used throughout the other notebooks in order to get to results
    * Code can be adjusted in these files in order to enhanced feature mining

