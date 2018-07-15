Neural Turing Machines (Pytorch)
=================================

Code for the Paper

**[Neural Turing Machines][1]**
Alex Graves, Greg Wayne, Ivo Danihelka


[1]: https://arxiv.org/abs/1410.5401

This repository contains code implemented for training, evaluating and visualizing results for the above mentioned paper from DeepMind.

Setup
=================================
The code is implemented in Pytorch 0.4. To setup, proceed as follows :

```
git clone https://www.github.com/kdexd/ntm-pytorch

```
The other python libraries that you'll need to use the code :
```
pip install numpy 
pip install tensorboard_logger
```

Training
================================
Training works with default arguments by :
```
python train.py
```
The script runs with all arguments set to default value. If you wish to changes any of these, run the script with ```-h``` to see available arguments and change them as per need be.

Evaluation
===============================
Evaluation can be done as follows :
```
python evaluate.py
```
Visualization
===============================
We have integrated Tensorboard_logger to visualize training and evaluation loss and bits per sequence error. To install tensorboard logger use :
```
pip install tensorboard_logger
```
Results
===============================
TBD 

Acknowledgements
===============================
TBD
