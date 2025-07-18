# Latent variable models

These tutorials will teach you about two popular latent variable models: mixture models and hidden Markov models. To fit these to data, you will use the expectation-maximization algorithm (EM). Tutorials are structured in the following way:

1) Mixture models and the general expectation maximization algorithm (2h, `mixture-models-EM.ipynb`)
   [![instructions](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bambschool/BAMB2025/blob/main/Day_4_latent_variable_models/mixture-models-EM.ipynb)
   [![solutions](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bambschool/BAMB2025/blob/main/Day_4_latent_variable_models/mixture-models-EM_solutions.ipynb)

3) Hidden Markov models (1.5h, `hidden-markov-models.ipynb`)

The hidden Markov model tutorial is accompanied by a library with pre-coded functions (`hmm_library.py`) that are partly based on code from the Linderman lab's [state space models repository](https://github.com/lindermanlab/ssm).

Theoretical introductions and code implementations are mainly based on Christopher Bishop's ["Pattern Recognition and Machine Learning"](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (2006), especially chapters 9 and 13. Other great intros (a bit more explicit on the derivations) can be found on Greg Gundersen's blog: A [theoretical intro to the EM algorithm](https://gregorygundersen.com/blog/2019/11/10/em/) and the [inference of posteriors in the HMM](https://gregorygundersen.com/blog/2020/11/28/hmms/).

Author (tutorials): Heike Stein
