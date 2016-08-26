# Painter by Numbers

Train and evaluate a model on whether or not two paintings are by the same
painter for the [Painter by Numbers challenge on Kaggle][Kaggle Challenge].

Uses ResNet50 to learn a distance metric model for whether two paintings are
the same.

Preprocessed data used to train and evaluate this model can also be found under
the [releases][Releases] tab.

## Todo:

* Better data augmentation/preprocessing?
    * Centering/whitening, rotations, etc.
* Use a triplet loss function, as described in [this paper][FaceNet].

[FaceNet]:          https://arxiv.org/abs/1503.03832
[Kaggle Challenge]: https://www.kaggle.com/c/painter-by-number
[Releases]:         https://github.com/zo7/painter-by-numbers/releases

