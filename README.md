# MFAN
  Code for the IJCAI-ECAI 2022 paper [MFAN: Multi-modal Feature-enhanced Attention Networks for Rumor Detection](https://www.ijcai.org/proceedings/2022/0335.pdf)
# Dataset
  We provide the pre-processed data files used for our experiments in the "dataset" folder. The raw datasets can be respectively downloaded from (https://www.dropbox.com/s/q4qe8rrroabkenv/twitter_w2v.bin?dl=0) and (https://www.dropbox.com/s/pqz43ue0e11aexq/weibo_w2v.bin?dl=0).
# Dependencies
  Our code runs with the following packages installed:
  ```
    python 3.8
    torch 1.7.1 + cu10.2
    torch-geometric=1.6.3=pypi_0
    torch-scatter=2.0.5=pypi_0
    torch-sparse=0.6.8=pypi_0
    torchvision=0.8.2=pypi_0
  ```
 more detail about virtual environment can be found in 
 ```
 requirement.txt
 ```
 # Run
 Train and test
 ```
 cd ./graph_part
 python pheme_threemodal.py 
 ```
