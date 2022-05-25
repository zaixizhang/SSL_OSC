# SSL_OSC
Graph Self-Supervised Learning for Optoelectronic Properties of Organic Semiconductors
<div align=center><img src="https://github.com/zaixizhang/SSL_OSC/blob/main/ssl_osc.png" width="700"/></div>

* `GIN/` contains codes for pretraining and finetuning on equilibrium molecules with GIN model.
  * `GIN/pretrain_masking.py` is the code for SSL pretraining and `GIN/finetune.py`  is the code for finetuning.
* `SchNet/` contains codes for pretraining and finetuning on non-equilibrium molecules with SchNet.
  * `SchNet/train_ssl.py` is the code for SSL pretraining and `SchNet/finetune.py`  is the code for finetuning.
