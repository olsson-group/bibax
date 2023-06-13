# BIBAX
Code to reproduce the analysis done in the paper: "A design strategy for improving vaccines via abrogation of host receptor binding" by Ratswohl et al. in EJI.


## Prelimaries
Relies on deep mutational scanning data from Blooms lab available here:

- Deep Mutational Scanning data from https://doi.org/10.1016/j.cell.2020.08.012 as `Deep_Mutational_Scan_RBD_stability_ACE2binding.csv`

- Anti-body escape data from https://doi.org/10.1016/j.chom.2020.11.007 as `Tableofmutation_antibody-escape_fraction_scores.csv`

## Setup and run
Using `conda` the we can set up an environment using
`conda create -n bibax python=3.7 numpy scipy matplotlib pandas`

We can then activate the environment and run the analysis
`conda activate bibax`

`python analysis-pipeline.py`
