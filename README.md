**NOTE: This doesn't work at the moment! I'll leave it up because the Releases section has a smaller dataset that I've made available, but don't bother looking at the code.**

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

[FaceNet]:          https://arxiv.org/abs/1503.03832
[Kaggle Challenge]: https://www.kaggle.com/c/painter-by-number
[Releases]:         https://github.com/zo7/painter-by-numbers/releases

