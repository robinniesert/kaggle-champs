# CHAMPS Predicting Molecular Properties (6th Place Solution)
Repository contains the source code for training the main model used to get 6th place: the molecular Transformer model with message passing layers. A more detailed description of the model is posted here: https://www.kaggle.com/c/champs-scalar-coupling/discussion/106407#latest-614111.

To get the processed data used for training, create a 'proc_data' folder and run ```preprocess.py``` and ```create_crossfolds.py``` in that order. Example usage for training in distributed mode on 2 GPUs: 
```
python -m torch.distributed.launch --nproc_per_node=2 train.py
```
