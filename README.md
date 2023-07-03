# Convert NII Data to Neuroglancer Precomputed Format Data

2023-06-13

## Setup

```shell
conda create -n nii2precomputed python=3.10
conda activate nii2precomputed
conda install zimg scikit-image numpy Pillow rich typer opencv -c fenglab
pip install tensorstore

# optional
conda install jupyter matplotlib black[jupyter] isort
```