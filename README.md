# Automatic augmentation by hill climbing

When doing data augmentation for images, it is not always intuitive for the user how much shear or translation to apply.

In this proposal, the model is always being trained in two branches, one branch having more intensive augmentation than the other.

After a period (e.g. a epoch), the two branches are evaluated using the validation. The weaker branch is eliminated, and the stronger branch becomes the parent of the following two branches.

* Ricardo Cruz, Joaquim F. Pinto Costa, and Jaime S. Cardoso. "Automatic Augmentation by Hill Climbing." International Conference on Artificial Neural Networks. Springer, Cham, 2019. **(to appear)**

The method is fairly easy to implement. In any case, this is how we implemented it in Keras. We saved and loaded models from the disk as necessary.

The code implements both a classifier and an U-Net architecture (`mymodels.py`) and uses our method in `train.py` and random and Bayesian search is implemented in `train_random.py`.

The data used is referenced in the paper. Please [email me](mailto:ricardo.pdm.cruz@gmail.com) if you have problems with anything.
