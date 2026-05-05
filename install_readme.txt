conda create -p .\env python=3.12 -y
conda activate .\env

conda activate H:\MyComicsTranslate\com-translate\env
pip install -r requirements.txt

"%ENV_PATH%\python.exe" -m pip install send2trash

for GPU
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
pip install D:\ocr\llama\llama_cpp_python-0.3.37+cu128.basic-cp312-cp312-win_amd64.whl
pip install ultralytics

////
conda env remove -p .\env #удаление 

conda env create -f env.yml -p .\env #создание с экспортом из файла

1)
///архивация окружения
conda activate .\env
conda pack -p .\env -o env-win-cu13.tar.gz

//восстановление окружения
mkdir env
tar -xzf env-win-cu13.tar.gz -C env

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

///////////////////////////
FOR LLAMA
pip uninstall torch torchvision torchaudio -y
pip uninstall llama-cpp-python -y
pip uninstall paddlepaddle-gpu -y

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130



pip install ultralytics