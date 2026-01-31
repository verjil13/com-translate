conda create -p .\env python=3.12 -y
conda activate .\env

conda activate H:\MyComicsTranslate\com-translate\env
pip install -r requirements.txt

for GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
conda install -c nvidia cuda-nvrtc=12.9.86 cuda=12.9 cudnn=9.14.0.64

Windows install CUDA_12.8, cudnn_9.8. 

////
conda env remove -p .\env #удаление 

conda env create -f env.yml -p .\env #создание с экспортом из файла

1)
///архивация окружения
conda activate .\env
conda pack -p .\env -o env-win-cu129.tar.gz

//восстановление окружения
mkdir env
tar -xzf env-win-cu129.tar.gz -C env

env\Scripts\activate
conda-unpack

2)
//////////////////////////
Дополнительно 
mkdir env.lock
conda list --explicit > env.lock/conda-win-cu129.lock
pip freeze > env.lock/requirements.lock.txt

//Восстановление
conda create -p .\env --file env.lock/conda-win-cu129.lock
conda activate .\env
pip install -r env.lock/requirements.lock.txt