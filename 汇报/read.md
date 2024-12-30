# Distribution-induced Bidirectional Generative Adversarial Network for Graph Representation Learning ([DBGAN](https://arxiv.org/abs/1912.01899))
  For an extended period, I have been focused on reproducing the code from the DBGAN paper. The original code from the paper was written in **TensorFlow** and is based on a very outdated version. Building on this, I have reimplemented the code using **PyTorch**. The main focus of this paper is graph representation learning, which involves mapping data into a low-dimensional space for representation. The entire process is illustrated in the diagram below.
<img src="/images/DBGAN.png" alt="vis" width="900"/>

## My work
I used the DBGAN proposed in the original paper as an upstream task to learn the features of bearing fault data (based on graph representation learning). For the downstream task, I created a very simple network structure aimed at performing fault diagnosis using the features learned from the upstream task.

## Deficiencies
Due to my limited capabilities, the reproduced code may still contain certain issues, resulting in a final fault diagnosis accuracy of only 90%. Below is the training loss during the DBGAN training process:
<img src="/images/loss.png" alt="vis" width="900"/>
For the **DBGAN code**, please refer to [DBGAN](optim_wgan.py).You can train DBGAN by running the following commands:

````
python optim_wgan.py
````
## Train
Run `python cnn.py` to perform the downstream fault diagnosis task. The entire training process is shown below:
````
python cnn.py
````
````
epoch: 9, train_loss: 1.3617, train_acc: 0.30, val_loss: 1.4309, val_acc: 0.0000
epoch: 19, train_loss: 1.3521, train_acc: 0.33, val_loss: 1.4246, val_acc: 0.1246
epoch: 29, train_loss: 1.3399, train_acc: 0.35, val_loss: 1.4094, val_acc: 0.1519
epoch: 39, train_loss: 1.3273, train_acc: 0.37, val_loss: 1.4100, val_acc: 0.1607
epoch: 49, train_loss: 1.3160, train_acc: 0.36, val_loss: 1.3993, val_acc: 0.1945
epoch: 59, train_loss: 1.3078, train_acc: 0.37, val_loss: 1.4049, val_acc: 0.1847
epoch: 69, train_loss: 1.2975, train_acc: 0.39, val_loss: 1.3940, val_acc: 0.2011
epoch: 79, train_loss: 1.2830, train_acc: 0.39, val_loss: 1.3942, val_acc: 0.2066
epoch: 89, train_loss: 1.2745, train_acc: 0.40, val_loss: 1.3879, val_acc: 0.2863
epoch: 99, train_loss: 1.2568, train_acc: 0.42, val_loss: 1.3830, val_acc: 0.2984
epoch: 109, train_loss: 1.2245, train_acc: 0.47, val_loss: 1.3826, val_acc: 0.3224
epoch: 119, train_loss: 1.2134, train_acc: 0.47, val_loss: 1.3787, val_acc: 0.3836
epoch: 129, train_loss: 1.1833, train_acc: 0.49, val_loss: 1.3565, val_acc: 0.4208
epoch: 139, train_loss: 1.1767, train_acc: 0.50, val_loss: 1.3413, val_acc: 0.4546
epoch: 149, train_loss: 1.1559, train_acc: 0.52, val_loss: 1.3234, val_acc: 0.4765
epoch: 159, train_loss: 1.1566, train_acc: 0.52, val_loss: 1.2994, val_acc: 0.4874
epoch: 169, train_loss: 1.1429, train_acc: 0.53, val_loss: 1.2866, val_acc: 0.4918
epoch: 179, train_loss: 1.1237, train_acc: 0.56, val_loss: 1.2574, val_acc: 0.4995
epoch: 189, train_loss: 1.1351, train_acc: 0.54, val_loss: 1.2509, val_acc: 0.5005
epoch: 199, train_loss: 1.1373, train_acc: 0.53, val_loss: 1.2398, val_acc: 0.5005
epoch: 209, train_loss: 1.1096, train_acc: 0.57, val_loss: 1.2391, val_acc: 0.5005
epoch: 219, train_loss: 1.1202, train_acc: 0.56, val_loss: 1.2312, val_acc: 0.5005
epoch: 229, train_loss: 1.1220, train_acc: 0.57, val_loss: 1.2283, val_acc: 0.5005
epoch: 239, train_loss: 1.1179, train_acc: 0.56, val_loss: 1.2218, val_acc: 0.5005
epoch: 249, train_loss: 1.1139, train_acc: 0.57, val_loss: 1.2227, val_acc: 0.5005
epoch: 259, train_loss: 1.1040, train_acc: 0.58, val_loss: 1.2169, val_acc: 0.5005
epoch: 269, train_loss: 1.1203, train_acc: 0.56, val_loss: 1.2121, val_acc: 0.5005
epoch: 279, train_loss: 1.1095, train_acc: 0.56, val_loss: 1.2076, val_acc: 0.5005
epoch: 289, train_loss: 1.1047, train_acc: 0.57, val_loss: 1.1974, val_acc: 0.5005
epoch: 299, train_loss: 1.0997, train_acc: 0.58, val_loss: 1.1994, val_acc: 0.5005
epoch: 309, train_loss: 1.1114, train_acc: 0.56, val_loss: 1.1957, val_acc: 0.4995
epoch: 319, train_loss: 1.0992, train_acc: 0.58, val_loss: 1.1795, val_acc: 0.4995
epoch: 329, train_loss: 1.1061, train_acc: 0.57, val_loss: 1.1813, val_acc: 0.5016
epoch: 339, train_loss: 1.1115, train_acc: 0.56, val_loss: 1.1694, val_acc: 0.5027
epoch: 349, train_loss: 1.1026, train_acc: 0.57, val_loss: 1.1390, val_acc: 0.5443
epoch: 359, train_loss: 1.1024, train_acc: 0.58, val_loss: 1.1358, val_acc: 0.5486
epoch: 369, train_loss: 1.1128, train_acc: 0.57, val_loss: 1.1188, val_acc: 0.6011
epoch: 379, train_loss: 1.1023, train_acc: 0.57, val_loss: 1.0846, val_acc: 0.7202
epoch: 389, train_loss: 1.0985, train_acc: 0.59, val_loss: 1.0729, val_acc: 0.7519
epoch: 399, train_loss: 1.0920, train_acc: 0.60, val_loss: 1.0333, val_acc: 0.8678
epoch: 409, train_loss: 1.0841, train_acc: 0.60, val_loss: 1.0326, val_acc: 0.8678
epoch: 419, train_loss: 1.0762, train_acc: 0.60, val_loss: 1.0416, val_acc: 0.8612
epoch: 429, train_loss: 1.0763, train_acc: 0.62, val_loss: 1.0362, val_acc: 0.8689
epoch: 439, train_loss: 1.0789, train_acc: 0.60, val_loss: 1.0104, val_acc: 0.9563
epoch: 449, train_loss: 1.0754, train_acc: 0.62, val_loss: 0.9970, val_acc: 0.9617
epoch: 459, train_loss: 1.0812, train_acc: 0.61, val_loss: 1.0014, val_acc: 0.9464
epoch: 469, train_loss: 1.0652, train_acc: 0.62, val_loss: 0.9840, val_acc: 0.9727
epoch: 479, train_loss: 1.0725, train_acc: 0.61, val_loss: 0.9822, val_acc: 0.9781
epoch: 489, train_loss: 1.0695, train_acc: 0.63, val_loss: 0.9894, val_acc: 0.9727
epoch: 499, train_loss: 1.0844, train_acc: 0.61, val_loss: 0.9739, val_acc: 0.9781
epoch: 509, train_loss: 1.0702, train_acc: 0.63, val_loss: 0.9704, val_acc: 0.9781
epoch: 519, train_loss: 1.0652, train_acc: 0.62, val_loss: 0.9842, val_acc: 0.9716
epoch: 529, train_loss: 1.0814, train_acc: 0.61, val_loss: 0.9776, val_acc: 0.9803
epoch: 539, train_loss: 1.0717, train_acc: 0.60, val_loss: 0.9817, val_acc: 0.9694
epoch: 549, train_loss: 1.0736, train_acc: 0.61, val_loss: 0.9843, val_acc: 0.9639
epoch: 559, train_loss: 1.0673, train_acc: 0.61, val_loss: 0.9712, val_acc: 0.9705
epoch: 569, train_loss: 1.0713, train_acc: 0.62, val_loss: 0.9733, val_acc: 0.9770
epoch: 579, train_loss: 1.0672, train_acc: 0.62, val_loss: 0.9777, val_acc: 0.9497
epoch: 589, train_loss: 1.0666, train_acc: 0.62, val_loss: 0.9944, val_acc: 0.9388
ACC: 0.9737704918032787, F1_Score: 0.9737704918032787
````
## Question
Both the training of DBGAN and the downstream tasks are extremely unstable. Therefore, everyone is welcome to refer to my code and engage in discussions. Let’s work together to figure out where these issues might be arising!


