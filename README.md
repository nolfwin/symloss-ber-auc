# On Symmetric Losses for Learning from Corrupted Labels [ICML'19]
Code for the paper: On Symmetric Losses for Learning from Corrupted Labels
Authors: Nontawat Charoenphakdee, Jongyeong Lee, Masashi Sugiyama

Implementation for experiments with Cifar-10 and Mnist
Paper link: [ArXiv](https://arxiv.org/abs/1901.09314)

## Usage

* install dependency : Python 3.5+, Pytorch 1.0, numpy, PIL, sklearn.
* run `python main.py --data cifar-10 --epoch 50 --prior 0.65 --opt auc`, which is default.
* data : Mnist (Odd vs Even), Cifar-10 (Airplane vs Horse)
* prior : one of (1.0, 0.0), (0.8, 0.3), (0.7, 0.4), (0.65, 0.45)
* opt : BER (balanced error rate minimization), AUC (area under the receiver operating characteristic curve maximization)


## Reference

[1] Nontawat Charoenphakdee, Jongyeong Lee, and Masashi Sugiyama.
"On Symmetric Losses for Learning from Corrupted Labels." In Proceedings of 36th International Conference on Machine Learning (ICML2019), Proceedings of Machine Learning Research, Long Beach, California, USA, Jun. 9-15, 2019.
