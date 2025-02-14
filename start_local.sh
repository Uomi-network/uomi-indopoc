git clone https://github.com/Uomi-network/uomi-indopoc
cd uomi-indopoc

conda create -n uomi-indopoc python=3.10 -y
conda activate uomi-indopoc

pip install torch
pip install redis
pip install autoawq

python main.py