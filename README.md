# Depression-In-Tweets

## Prerequisites
* Python 3
* CPU or NVIDIA GPU + CUDA CuDNN

## Getting started
* Please Install PyTorch and the other relevant dependencies.
* Please run the command `pip install -r requirements.txt.`

## Dataset
This data was collected from the paper titled ```"DEPTWEET: A Typology for Social Media Texts to Detect Depression Severities" published in the journal of Computers in Human Behavior (Q1, H-Index: 226, Impact Factor: ~10).```
The data folder contains tweet IDs corresponding to severities of depression collected from Twitter. Owing to Twitter's policy on data sharing, we can not directly upload the dataset and are only restricted to sharing tweet-id's here. 


* Go to the Scripts directory:
```cd Scripts```

* Place the dataset in the ../Dataset/ folder and run the command:
```python dataset.py```

## Training and Evaluation

* Go to the Scripts directory:
```cd Scripts```

* Train stand-alone model with GPU support:
```python train.py --pretrained_model "albert-base-v2" --epochs 10```


* Information regarding other training parameters can be found at `Scripts/common.py file.`

* Fine-tuned models will be saved at `../Models/` folder.

* Evaluation output files will be saved at `../Output/` folder.

* Figures will be saved at `../Figures/` folder.


