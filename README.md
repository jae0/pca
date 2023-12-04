# PCA, pPCA in Julia/Turing

(Probabilistic, Latent) Principle Components Analysis Bayesian Householder Rotation invariant form.

Copied and adapted from the following sources (all credit there), especially:

[Rajbir S. Nirwan and Nils Bertschinger, 2019. Proceedings of the 36 th International Conference on Machine Learning, Long Beach, California, PMLR 97](https://github.com/jae0/HouseholderBPCA/blob/master/py_stan_code/ppca_house_improved.stan)
 
And the [main Julia/Turing pPCA tutorial](https://turing.ml/v0.21/tutorials/11-probabilistic-pca) had done the brunt of the conversion from python/Stan.

Minor changes and adaptation for my own needs ([species composition in marine environments and feeding networks](https://github.com/jae0/aegis)).

Example work flow: see [pca.md](pca.md) 

Or to use directly from R (you will need to install julia and required libs on your own): [pca_from_R.md](pca_from_R.md)
