# L-NeuralODE


This package implements the latent space forecasting of the Dynamical Autoencoder (DnAE) framework, which combines convolutional autoencoders with parameterized Neural ODEs to forecast complex, high-dimensional spatiotemporal dynamics. 

To get started, the user may go to example_cases/Lorenz96 and sequentially (in numerical order) run scripts 00_,01_,02_ to generate the data, train the neural ODE and run predictions, and finally evaluate performance. This includes an example of the curriculum learning procedure.

The convolutional Autoencoder to spatially compress fields is documented in:  Saetta E, Tognaccini R, Iaccarino G. Uncertainty quantification in autoencoders predictions: Applications in aerodynamics. Journal of Computational Physics. 2024
