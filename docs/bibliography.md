# Bibliography (papers & websites) and resources 

## Conferences 

* [Dmitry Larko, H2O.ai - Kaggle Airbus Ship Detection Challenge - H2O World San Francisco](https://www.youtube.com/watch?v=0Opb8gB1p4w)

* [(Ru) Airbus Ship Detection Challenge, 4th position by Dmitry Danevskyi @ ML #5](https://www.youtube.com/watch?v=pY3HaHFB7yA)

## Libraries

* [Albumentations](https://github.com/albumentations-team/albumentations)

* [Segmentation Models](https://github.com/qubvel/segmentation_models)

## General information

* [Limits to visual representational correspondence between convolutional neural networks and the human brain](https://www.nature.com/articles/s41467-021-22244-7#Sec7)

* [An Evaluation of Deep Learning Methods for Small Object Detection](https://www.hindawi.com/journals/jece/2020/3189691/)

* [Review of deep learning: concepts, CNN architectures, challenges, applications, future directions](https://www.researchgate.net/publication/350527503_Review_of_deep_learning_concepts_CNN_architectures_challenges_applications_future_directions)

* [Resources for performing deep learning on satellite imagery](https://github.com/robmarkcole/satellite-image-deep-learning#object-detection-enhanced-by-super-resolution)

## Training strategy and hyperparameters
* [The Million-Dollar Question: When to Stop Training your Deep Learning Model](https://towardsdatascience.com/the-million-dollar-question-when-to-stop-training-deep-learning-models-fa9b488ac04d)
\
**Notes**: \
At a certain time, the validation error will start to flatten out or increase
validation increasing → overfitting (not necessarily) 
triggers to stop training: \
→ the precision metric has not increased for several epochs \
→ the precision metric has not increased by more than a certain delta and is therefore not considered an improvement \
→ early stopping patience: the accuracy has not improved over the best results in the last X epochs

* [Revisiting Small Batch Training for Deep Neural Network (paper)](https://arxiv.org/pdf/1804.07612.pdf) \
**Notes:** \
→ Linear scaling rule:  when increasing the batch size, a linear increase of the learning rate η with the batch size m is required to keep the mean SGD weight update per training example constant.  \
“The purely formal difference of using the average of the local gradients instead of the sum has
favoured the conclusion that using a larger batch size could provide more ‘accurate’ gradient estimates and allow the use of larger learning rates. However, the above analysis shows that, from the
perspective of maintaining the expected value of the weight update per unit cost of computation, this
may not be true. In fact, using smaller batch sizes allows gradients based on more up-to-date weights
to be calculated, which in turn allows the use of higher base learning rates, as each SGD update has
lower variance. Both of these factors potentially allow for faster and more robust convergence” 

    → Using sum rather than mean in the SGD weight update:
“under the assumption of constant η, large batch training
can be considered to be an approximation of small batch methods that trades increased parallelism
for stale gradients”

    → effect of batch normalization : \
    “For very small batches, the estimation of the batch mean and variance can be very noisy, which may limit the effectiveness of BN in reducing the covariate shift”
“ with very small batch sizes the estimates of the batch mean and variance
used during training become a less accurate approximation of the mean and variance used for testing” \
“Hoffer et al. (2017) have shown empirically that the reduced generalization performance of large batch training is connected to the reduced number of parameter updates over the same number of epochs (which corresponds to the same computation cost in number of gradient calculations). Hoffer et al. (2017) present evidence that it is possible to achieve the same generalization performance with large batch size, by increasing the training duration to perform the same number of SGD updates”

    → “For all the results, the reported test or validation accuracy is the median of the final 5 epochs of training (Goyal et al., 2017)” 

    → performances: The results show a clear performance improvement for progressively smaller batch sizes.

* [An overview of gradient descent optimization algorithms (paper)](https://arxiv.org/pdf/1609.04747.pdf) \
**Notes:** \
3 gradient descent variants: batch (aka vanilla), stochastic (aka SGD), mini-batch (also called SGD) \
4 challenges : \
→ Choosing a proper learning rate \
→ Learning rate schedule… have to be defined in advance and are thus unable to adapt to a dataset’s characteristics \
→ The same learning rate applies to all parameter updates \
→ “Another key challenge of minimizing highly non-convex error functions common for neural networks is avoiding getting trapped in their numerous suboptimal local minima. Dauphin et al. [ 5 ] argue that the difficulty arises in fact not from local minima but from saddle points, i.e. points where one dimension slopes up and another slopes down. These saddle points are usually surrounded by a plateau of the same error, which makes it notoriously hard for SGD to escape, as the gradient is close to zero in all dimensions”

    Gradient descent optimization algorithms : \
→ Momentum : take into account what happened before \
→ Nesterov Accelerated Gradient (NAG) : take into account what happened before and what will happen \
→ Adagrad : use a different learning rate for every parameter at every step

    “Insofar, Adam might be the best overall choice”

* [Transfer learning: the dos and don’ts](https://medium.com/starschema-blog/transfer-learning-the-dos-and-donts-165729d66625) \
**Notes:** \
“The biggest benefit of transfer learning shows when the target data set is relatively small”

    “When transferring to a task with a relatively large data set, you might want to train the network from scratch (which would make it not transfer learning at all). At the same time — given that such a network would be initialized with random values, you have nothing to lose by using the pretrained weights! Unfreeze the entire network, remove the output layer and replace it with one matching the number of destination task classes, and fine-tune the whole network.”

* [Hyperparameter optimization](https://nanonets.com/blog/hyperparameter-optimization/) \
**Notes:** \
Hyperparameter Optimization Algorithms : \
→ Grid search \
→ Random search \
→ Bayesian Optimization

* [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay (paper)](https://arxiv.org/abs/1803.09820) \
**Notes:** \
“Specifically, this report shows how to examine the training validation/test loss function for subtle clues of underfitting and overfitting and suggests guidelines for moving toward the optimal balance point. Then it discusses how to increase/decrease the learning rate/momentum to speed up training”
    
    “The conventional method is to perform a grid or a random search, which can be
computationally expensive and time consuming”
    
    Cyclical learning rates (CLR), LR range test = a recipe for choosing the learning rate

* [On Hyperparameter Optimization of Machine Learning Algorithms: Theory and Practice (paper)](https://arxiv.org/pdf/2003.05689.pdf)

* [Why, How and When to Scale your Features](https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e)

* [Normalization vs Standardization — Quantitative analysis](https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf)

## Data augmentation

* [Augmentation for small object detection (paper)](https://arxiv.org/pdf/1902.07296.pdf)

* [Learning Data Augmentation Strategies for Object Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720562.pdf) \
**Notes:** \
"Clearly we see that changing augmentation can be as, if not more,
powerful than changing around the underlying architectural components."

