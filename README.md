# MFAN
  Code for the IJCAI-ECAI 2022 paper [MFAN: Multi-modal Feature-enhanced Attention Networks for Rumor Detection](https://www.ijcai.org/proceedings/2022/0335.pdf)
# Dataset
  The datasets used in the experiments were based on the two publicly available Weibo and PHEME datasets released by Song et al. (2019) and Zubiaga et al. (2017), and the preprocess process was based on the work by Yuan et al. (2019):
  
  Changhe Song, Cheng Yang, Huimin Chen, Cunchao Tu, Zhiyuan Liu, and Maosong Sun. Ced: Credible early detection of social media rumors. IEEE Transactions on Knowledge and Data Engineering, 33(8):3035–3047, 2019.
  
  Arkaitz Zubiaga, Maria Liakata, and Rob Procter. Exploiting context for rumour detection in social media. InInternational Conference on Social Informatics, pages 109–123. Springer, 2017.
  
  Chunyuan Yuan, Qianwen Ma, Wei Zhou, Jizhong Han, and Songlin Hu. Jointly embedding the local and global relations of heterogeneous graph for rumor detection. InICDM, pages 796–805. IEEE, 2019.
  
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
    numpy 1.23.0
    pandas 1.3.0
    scikit-learn 1.0.2
    gensim 3.7.2
    jieba 0.39
 ```
 # Run
 Train and test
 ```
 cd ./graph_part
 python pheme_threemodal.py 
 ```
