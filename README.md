Neural Turing Machines (Pytorch)
=================================
Code for the paper

**[Neural Turing Machines][1]**

Alex Graves, Greg Wayne, Ivo Danihelka

[1]: https://arxiv.org/abs/1410.5401
Neural Turing Machines (NTMs) contain a recurrent network coupled with an external memory resource, which it can interact with by attentional processes. Therefore NTMs can be called Memory Augmented Neural Networks. They are end-to-end differentiable and thus are hypothesised at being able to learn simple algorithms. They outperform LSTMs in learning several algorithmic tasks due to the presence of external memory without an increase in parameters and computation.

This repository is a stable Pytorch implementation of a Neural Turing Machine and contains the code for training, evaluating and visualizing results for the Copy, Repeat Copy, Associative Recall and Priority Sort tasks. The code has been tested for all 4 tasks and the results obtained are in accordance with the results mentioned in the paper. The training and evaluation code for N-Gram task has been provided however the results would be uploaded after testing.

<p align="center">
<img width="500" height="300" src="https://www.researchgate.net/profile/Gabriel_Makdah/publication/279864730/figure/fig3/AS:372237233344513@1465759680918/Neural-Turing-Machine-architecture-The-controller-or-neural-network-receives-the-input.png">
</p>
<p align="center">
<em> Neural Turing Machine architecture </em>
</p>

Setup
=================================
Our code is implemented in Pytorch 0.4.0 and Python >=3.5. To setup, proceed as follows :

To install Pytorch head over to ```https://pytorch.org/``` or install using miniconda or anaconda package by running 
```conda install -c soumith pytorch ```.

Clone this repository :

```
git clone https://www.github.com/kdexd/ntm-pytorch
```

The other python libraries that you'll need to run the code :
```
pip install numpy 
pip install tensorboard_logger
pip install matplotlib
pip install tqdm
pip install Pillow
```

Training
================================
Training works with default arguments by :
```
python train.py
```
The script runs with all arguments set to default value. If you wish to changes any of these, run the script with ```-h``` to see available arguments and change them as per need be.
```
usage : train.py [-h] [-task_json TASK_JSON] [-batch_size BATCH_SIZE]
                [-num_iters NUM_ITERS] [-lr LR] [-momentum MOMENTUM]
                [-alpha ALPHA] [-saved_model SAVED_MODEL] [-beta1 BETA1] [-beta2 BETA2]
```
Both RMSprop and Adam optimizers have been provided. ```-momentum``` and ```-alpha``` are parameters for RMSprop and ```-beta1``` and ```-beta2``` are parameters for Adam. All these arguments are initialized to their default values.

The smoothing factor for all curves is ```0.6```
- Training for copy task is carried out with sequence length ranging from 1-20. The curve for bits per sequence error vs iterations for this task is shown below :
![Alt text](https://github.com/vlgiitr/ntm-pytorch/blob/master/images/copy_loss.png)

- Training for repeat copy task is carried out with sequence length ranging from 1-10 and repeat number in the range 1-10. The curve for bits per sequence error vs iterations for this task is shown below :
![Alt text](https://github.com/vlgiitr/ntm-pytorch/blob/master/images/repeat_copy_loss.png)

- Training for associative recall task is carried out the number of items ranging from 2-6.The curve for bits per sequence error vs iterations for this task is shown below :
![Alt text](https://github.com/vlgiitr/ntm-pytorch/blob/master/images/associative_loss.png)

- Training for priority sort task is carried outwith an input sequence length of 20 and target sequence length of 16. The curve for bits per sequence error vs iterations for this task is shown below :
![Alt text](https://github.com/vlgiitr/ntm-pytorch/blob/master/images/prioritysort_loss.png)


Evaluation
===============================
The model was trained and was evaluated as mentioned in the paper. The results were in accordance with the paper. Saved models for all the tasks are available in the ```saved_models``` folder. The model for copy task has been trained upto 500k iterations and those for repeat copy, associative recall and priority sort have been trained upto 100k iterations. The code for saving and loading the model has been incorporated in ```train.py``` and ```evaluate.py``` respectively.

The evaluation parameters for all tasks have been included in ```evaluate.py```.

Evaluation can be done as follows :
```
python evaluate.py
```
- Results for copy task shows that the NTM generalizes well for sequence length upto 120. The target and output for copy task is shown below :

![Alt text](https://github.com/vlgiitr/ntm-pytorch/blob/master/images/copy_1.png)

- Results for the repeat copy task shows that the NTM generalizes well for maximum sequence length of 20 and repeat number     upto 20. The target and output for repeat copy task is shown below :

![Alt text](https://github.com/vlgiitr/ntm-pytorch/blob/master/images/repeat_copy_seq_len_1.png)

- Results for associative recall task shows that the NTM generalizes well for number of items upto 20. The target and output for associative recall task is shown below :

![Alt text](https://github.com/vlgiitr/ntm-pytorch/blob/master/images/associative_2.png)

- Results for the priority sort task also show the better generalization capability of the NTM. The target and output for priority sort task is shown below :

![Alt text](https://github.com/vlgiitr/ntm-pytorch/blob/master/images/priority_sort_1.png)


Visualization
===============================
We have integrated Tensorboard_logger to visualize training and evaluation loss and bits per sequence error. To install tensorboard logger use :
```
pip install tensorboard_logger
```

Sample outputs and bits per sequence error curves have been provided in the ```images``` folder.

Acknowledgements
===============================
- We have used the following codebase as a reference for our implementation : **[loudinthecloud/pytorch-ntm][2]**  

[2]:https://github.com/loudinthecloud/pytorch-ntm

LICENSE
===============================
MIT
