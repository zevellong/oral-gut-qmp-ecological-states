# Oral-gut QMP ecological states

This repository contains custom Python scripts used to generate the main figures for the manuscript:

**“Quantitative microbiome profiling resolves distinct ecological states of oral-derived bacteria in the gut.”**

## Repository scope

This repository provides downstream figure-generation scripts for the main analyses presented in the manuscript. The scripts are intended to document how the main figures were generated from processed metagenomic input tables, including species-level relative abundance profiles, predicted fecal microbial load, sample metadata and oral-derived species definitions.

This repository includes:

* scripts for generating the five main figures;
* shared plotting configuration;
* shared data-loading and utility functions;
* the conda environment file used for the analysis.

This repository does not include:

* raw sequencing data;
* large processed metagenomic matrices;
* intermediate cache files;
* temporary notebooks;
* exploratory analysis files.

Raw metagenomic sequencing data analyzed in the study are publicly available from the original repositories listed in the manuscript and Supplementary Tables.

## Repository structure

```text
oral-gut-qmp-ecological-states/
├── README.md
├── environment.yml
├── fig1.py
├── fig2.py
├── fig3.py
├── fig4.py
├── fig5.py
├── proj_config.py
└── proj_plot_main_func.py
```

## Main figure scripts

The five main figure scripts are:

```text
fig1.py    Representative ecological states under RMP and QMP
fig2.py    Pan-disease QMP landscape
fig3.py    Species-level decomposition of oral-derived signatures
fig4.py    Antibiotic perturbation dynamics
fig5.py    Microbial load versus oral RMP regression analysis
```

Shared configuration and utility functions are provided in:

```text
proj_config.py
proj_plot_main_func.py
```

`proj_config.py` contains shared plotting settings and project-level path configuration.
`proj_plot_main_func.py` contains shared data-loading and preprocessing functions used by the figure scripts.

## Required input data

The scripts are designed to run from processed input tables generated according to the Methods section of the manuscript.

Required processed inputs include:

```text
sample metadata table
species-level relative abundance table
predicted fecal microbial load table
oral-derived species catalog
cohort inclusion and exclusion information
```

Large processed matrices and intermediate cache files are not included in this repository. These files are derived from publicly available metagenomic datasets and from the processing workflow described in the manuscript.

## Oral-derived and gut fractions

Oral-derived taxa were defined using the consensus oral-derived catalog described in the manuscript. Oral RMP was calculated as the summed relative abundance of species in this consensus oral-derived catalog, whereas gut RMP was calculated as the remaining gut fraction after excluding the consensus oral-derived taxa.

Species-level QMP values were calculated by multiplying species-level relative abundance by predicted total microbial load. Based on the oral and gut fractions, oral QMP was calculated as the summed species-level QMP values of taxa in the consensus oral-derived catalog, whereas gut QMP was calculated as the summed QMP values of the remaining gut fraction.

## Software environment

The analysis was performed in a conda environment. The environment can be created using:

```bash
conda env create -f environment.yml
conda activate zwml
```

If the environment already exists, it can be updated using:

```bash
conda env update -f environment.yml --prune
```

Some packages were installed through pip within the conda environment, as specified in `environment.yml`.

## Notes on reproducibility

This repository is not intended to provide a raw FASTQ-to-figure pipeline. Instead, it documents the downstream analysis and visualization code used to generate the main figures from processed metagenomic profiles and predicted microbial-load tables.

Users wishing to reproduce the full analysis should first obtain the public metagenomic datasets listed in the manuscript, process them according to the Methods section, and then provide the processed input tables required by the figure-generation scripts.

## Contact

For questions about the code or analysis, please contact:

Wei-Hua Chen
[weihuachen@hust.edu.cn](mailto:weihuachen@hust.edu.cn)
