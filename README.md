# TMU System for SLAM-2018
  This code is [TMU System](http://sharedtask.duolingo.com/papers/kaneko.slam18.pdf) for [SLAM-2018](http://sharedtask.duolingo.com/)
### Requirements
  - python==3.6.0
  - torch==0.4.1
  - gensim==3.6.0
### How to use
  You need to download data from [this site](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8SWHNO) and place them to data directory.
  Preprocessing the data for model training:
  ```sh
  $ mkdir model_data
  $ python preprocessRNNLM.py -train_data data/en_es.slam.20171218.train -valid_data en_es.slam.20171218.dev -valid_key en_es.slam.20171218.dev.key -save_data model_data/en_es
  ```
  Training the model:
  ```sh
  $ python trainRNNLM.py -data model_data/en_es.train.pt -save_model model/enes -gpus 0
  ```
  Testing the model:
  ```sh
  $ python test.py -model model/[model name] -data data/en_es.slam.20171218.dev -output enes.pre
  ```
### Papers
  When referencing this code, please cite [this paper](http://sharedtask.duolingo.com/papers/kaneko.slam18.pdf).
  ```
  @InProceedings{W18-0544,
    author =  "Kaneko, Masahiro
      and Kajiwara, Tomoyuki
      and Komachi, Mamoru",
    title =  "TMU System for SLAM-2018",
    booktitle =  "Proceedings of the Thirteenth Workshop on Innovative Use of NLP for Building Educational Applications",
    year =  "2018",
    publisher =  "Association for Computational Linguistics",
    pages =  "365--369",
    location =  "New Orleans, Louisiana",
    url =  "http://aclweb.org/anthology/W18-0544"
  }
  ```
