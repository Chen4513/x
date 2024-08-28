cd students-coordination/
conda env create -f environment.yml
conda activate rllib_2.2
cd environment/multigrid/
pip install -e .
cd ../..
pip install "pydantic<2"