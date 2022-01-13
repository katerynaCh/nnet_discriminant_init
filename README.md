# Neural Network initialization based on discriminant learning
This repository provides the implementation for the paper "[Feedforward neural networks initialization based on discriminant learning]". The initialization scripts for MLPs and CNNs are provided along with the BatchNormalization layer described in the paper.

# Running the code
We provide initialization and training scripts for MNIST and CIFAR10 datasets. To perform initialization of an MLP architecture run
```
python init_mlp.py
```
Inside the script you should specify the dataset, activation function, and other hyperparameters that are self-explanatory from the names. Similarly, to initialize a CNN architecture run (with desired parameters)
```
python init_cnn.py
```
Either of those scripts will create a pickle file containing the weights of corresponding layers, first to last. The pickle file can be used for weight initalization in the main training loop. We provide an example for 16-subclass architecture (i.e., with first layer having 159 neurons/filters) that builds the architecture and initializes it with weights from a given file. For MLP, specify in the scipt the dataset and run:
```
python train_mlp.py
```
For CNN:
```
python train_cnn.py
```

# Environment
The code is built with Tensorflow 2. 

# Citing the work
If you find our work useful, kindly cite as:
```
@article{chumachenko2022feedforward,
  title={Feedforward neural networks initialization based on discriminant learning},
  author={Chumachenko, Kateryna and Iosifidis, Alexandros and Gabbouj, Moncef},
  journal={Neural Networks},
  volume={146},
  pages={220--229},
  year={2022},
  publisher={Elsevier}
}
```
[Feedforward neural networks initialization based on discriminant learning]: <https://www.sciencedirect.com/science/article/pii/S0893608021004482>
