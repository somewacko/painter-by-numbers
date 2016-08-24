# Painter by Numbers

Train and evaluate a model on whether or not two paintings are by the same
painter for the [Painter by Numbers challenge on Kaggle][Kaggle Challenge].

Uses a ResNet50 model to learn a distance metric for whether two paintings are
the same using a triplet loss function, which compares an example image with
both a positive and a negative case and tries to enforce a margin between them.

[Kaggle Challenge]: https://www.kaggle.com/c/painter-by-number
