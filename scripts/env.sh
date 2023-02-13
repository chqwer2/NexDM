https://github.com/chqwer2/NexDM/blob/main/scripts/env.sh
source ~/.bashrc
conda create -n "nex" python=3.8

pip install PyYAML

pip install -r stablediffusion/enviornment.yaml
pip install -r nex-code/enviornment.yaml

