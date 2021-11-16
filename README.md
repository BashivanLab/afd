
Official code for Adversarial Feature Desensitization (AFD). 

https://arxiv.org/abs/2006.04621

You can run training procedure by calling ```afd_train.py```. It currently supports MNIST, CIFAR10, and CIFAR100 datasets. 
We have tested the code with ResNet18 on MNIST, CIFAR10 and CIFAR100. 

**Example**: 
```
python afd_train.py --dataset=cifar10 --enc_model=resnet18norm --save_path=[SAVE_PATH]
```

### Model checkpoints: 
Download the pretrained models from the links below: 

[MNIST-checkpoint](https://drive.google.com/file/d/1-6bylNZ9ZgZ3DvwcRJ1nNCLbXDiw9E1D/view?usp=sharing)

[CIFAR10-checkpoint](https://drive.google.com/file/d/1---psLlEd9N4Kv13bpfQoNHnMOdtXbkc/view?usp=sharing)

[CIFAR100-checkpoint](https://drive.google.com/file/d/1-7rKQYxDU9JSz1vPgOPp7R5Qbe6hhCu4/view?usp=sharing)

Use ```notebooks/test.ipynb``` to run attacks on the pretrained models.  


# Reference
```
@inproceedings{bashivan2021adversarial,
  title={Adversarial Feature Desensitization},
  author={Bashivan, Pouya and Bayat, Reza and Ibrahim, Adam and Ahuja, Kartik and Faramarzi, Mojtaba and Laleh, Touraj and Richards, Blake and Rish, Irina},
  journal={arXiv preprint arXiv:2006.04621},
  booktitle={NeurIPS},
  year={2021}
}
```