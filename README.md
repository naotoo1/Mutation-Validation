# Mutation-Validation for Learning Vector Quantization



In this paper is posited a new model validation scheme for LVQ (Learning Vector Quantization) models. This repository contains all the implementaions of the mutation validation algorithm and models that were used for the evaluation. 

## Abstract
_Mutation validation as a complement to existing applied machine learning validation schemes has been explored in recent times. Exploratory work for Learning vector quantization (LVQ) based on this model validation scheme remains to be discovered. This paper proposes mutation validation as an extension to existing cross-validation and holdout schemes for Generalized LVQ and its advanced variants. The mutation validation scheme provides a responsive, interpretable, intuitive and easily comprehensible score that complements existing validation schemes employed in the performance evaluation of the prototype-based LVQ family of classification algorithms. This paper establishes a relation between the mutation validation scheme and the goodness of fit evaluation for four LVQ models: Generalized LVQ, Generalized Matrix LVQ, Generalized Tangent LVQ and Robust Soft LVQ models. Numerical analysis regarding model complexity and effects on test outcomes, pitches mutation validation scheme above cross-validation and holdout schemes_.


## About the models

The details of the implementation and results evaluation can be found in the paper. The results of MV, CV and Holdout schemes for two artificially generated datasets and two real word data sets(WDBC and MNIST) against increasing LVQ model(s) complexity are presented in the paper. The target space perturbation algorithm presented in the paper is not only limited for use in LVQ's but can also be used for any supervised machinelearning/deep learning model. The mutation validation scheme presented in this paper can also be adopted in parallel to existing machine learning evaluation pipelines in runtime.  
Privided below is a brief of the # of prototypes vs MV, CV and Holdout evaluation metric scores(accuracy). Higher scores mean good fit of the decision boundary of the model and vice-versa.
