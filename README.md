# SSL_OSC
Graph Self-Supervised Learning for Optoelectronic Properties of Organic Semiconductors (https://arxiv.org/abs/2112.01633)
<div align=center><img src="https://github.com/zaixizhang/SSL_OSC/blob/main/ssl_osc.png" width="700"/></div>

* `GIN/` contains codes for pretraining and finetuning on equilibrium molecules with GIN model.
  * `GIN/pretrain_masking.py` is the code for SSL pretraining and `GIN/finetune.py`  is the code for finetuning.
* `SchNet/` contains codes for pretraining and finetuning on non-equilibrium molecules with SchNet.
  * `SchNet/train_ssl.py` is the code for SSL pretraining and `SchNet/finetune.py`  is the code for finetuning.

## Cite

If you find this repo to be useful, please cite our paper. Thank you.

```
@article{zhang2021graph,
  title={Graph Self-Supervised Learning for Optoelectronic Properties of Organic Semiconductors},
  author={Zhang, Zaixi and Liu, Qi and Zhang, Shengyu and Hsieh, Chang-Yu and Shi, Liang and Lee, Chee-Kong},
  journal={arXiv preprint arXiv:2112.01633},
  year={2021}
}
```
