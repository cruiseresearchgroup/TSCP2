# <em>TS-CP<sup>2</sup></em>
### Time Series Change Point Detection based on Contrastive Predictive Coding

## Abstract
Change Point Detection techniques aim to capture changes in trends and sequences in time-series data to describe the underlying behaviour of the system.
Detecting changes and anomalies in the web services, the trend of applications usage can provide valuable insights into the system. However, many existing approaches are done in a supervised manner, requiring well-labelled data. As the amount of data produced and captured by sensors are growing rapidly, it is getting harder and even impossible to annotate the data. Therefore, coming up with a self-supervised solution is a necessity these days. 
In this work, we propose <em>TS-CP<sup>2</sup></em> a novel self-supervised technique for temporal change point detection, based on representation learning with a Temporal Convolutional Network (TCN). To the best of our knowledge, our proposed method is the first method which employs Contrastive Learning for prediction with the aim of change point detection.
Through extensive evaluations, we demonstrate that our method outperforms multiple state-of-the-art change point detection and anomaly detection baselines, including those adopting the either unsupervised or semi-supervised approach. <em>TS-CP<sup>2</sup></em> is shown to improve both non-Deep learning- and Deep learning-based methods by 0.28 and 0.12 in terms of average F1-score across three datasets.

Link to arXiv version [here](https://arxiv.org/abs/2011.14097)


## The script to run the model

    python 3 main.py  --datapath <path_to_dataset> 
                      --output <path_to_output> 
                      --win <window_size> 
                      --dataset <dataset_name> 
                      --batch <batch_size>  
                      --code <code_size> 
                      --sim [cosine||] 
                      --loss [nce|fc|dcl|harddcl] 
                      --lr [0.00005|0.0001] 
                      --temp 0.5 
                      --tau <> 
                      --beta <>

## Bibtex
If you find this code or the paper useful, please consider citing:

     
    @inproceedings{deldari2021tscp2,
    title={Time Series Change Point Detection with Self-Supervised Contrastive Predictive Coding}, 
    author={Deldari, Shohreh and Smith, Daniel V. and Xue, Hao and Salim, Flora D. },
    year = {2021},
    publisher = {Association for Computing Machinery},
    url = {https://doi.org/10.1145/3442381.3449903},
    doi = {10.1145/3442381.3449903},
    booktitle = {Proceedings of The Web Conference 2021},
    pages = {},
    numpages = {12},
    series = {WWW '21}
    }



    
