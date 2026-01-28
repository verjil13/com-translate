conda create -n nump_torch python=3.12 -y

conda create -p .\env python=3.12 -y
conda activate .\env

conda activate H:\com-translate\env
pip install -r requirements.txt

for GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Windows install CUDA_12.8, cudnn_9.8. 


conda activate H:\comic-translate\env

