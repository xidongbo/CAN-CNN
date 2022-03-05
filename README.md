# Reference
If you are interested in the code, please cite our paper:
```
Xi D, Zhuang F, Zhou G, et al. Domain adaptation with category attention network for deep sentiment analysis[C]//Proceedings of The Web Conference 2020. 2020: 3133-3139.
```
or in bibtex style:
```
@inproceedings{xi2020domain,
  title={Domain adaptation with category attention network for deep sentiment analysis},
  author={Xi, Dongbo and Zhuang, Fuzhen and Zhou, Ganbin and Cheng, Xiaohu and Lin, Fen and He, Qing},
  booktitle={Proceedings of The Web Conference 2020},
  pages={3133--3139},
  year={2020}
}
```

# CAN-CNN
Keras implementation of the Category Attention Network and Convolutional Neural Network based model (CAN-CNN).  
Code for the paper accepted by WWW20: 
Domain Adaptation with Category Attention Network for Deep Sentiment Analysis.
[https://arxiv.org/pdf/2112.15290.pdf]


# Example to run the model
```
python CAN-CNN.py -gpus 0 -dataset1 MR -dataset2 CR -model_name can-cnn.h5  -fold 5 -alpha 0.03 -beta 0.05 -gamma 0.01 -topk 5 -topn 50
```


Last Update Date: Mar. 5, 2022 (UTC+8)
