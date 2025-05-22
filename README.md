# Tuned-Thermal-Neural-Network

## Purpose
This repository demonstrates the application of thermal neural networks (TNNs) on an electric motor data set.

The data set is freely available at [Kaggle](https://www.kaggle.com/wkirgsn/electric-motor-temperature)


## Topology

A physics‑informed neural network that emulates classic lumped‑parameter thermal circuits, now extended with automatic hyper‑parameter tuning, post‑hoc explainability, and edge deployment via TensorFlow Lite.

Three function approximators (e.g., multi-layer perceptrons (MLPs)) model the thermal parameters (i.e., thermal conductances, thermal capacitances, and power losses) of an arbitrarily complex component arrangement in a system.
Such a system is assumed to be sufficiently representable by a system of ordinary differential equations (not partial differential equations!).

One function approximator outputs thermal conductances, another the inverse thermal capacitances, and the last one the power losses generated within the components.
Although thermal parameters are to be estimated, their ground truth is not required.
Instead, measured component temperatures can be plugged into a cost function, where they are compared with the estimated temperatures that result from the thermal parameters that are estimated from the current system excitation.
[Error backprop through time](https://en.wikipedia.org/wiki/Backpropagation_through_time) will take over from here. 

The TNN's inner cell working is that of [lumped-parameter thermal networks](https://en.wikipedia.org/wiki/Lumped-element_model#Thermal_systems) (LPTNs).
A LPTN is an electrically equivalent circuit whose parameters can be interpreted to be thermal parameters of a system.
A TNN can be interpreted as a hyper network that is parameterizing a LPTN, which in turn is iteratively solved for the current temperature prediction.

In contrast to other neural network architectures, a TNN needs at least to know which input features are temperatures and which are not.
Target features are always temperatures.

In a nutshell, a TNN solves the difficult-to-grasp nonlinearity and scheduling-vector-dependency in [quasi-LPV](https://en.wikipedia.org/wiki/Linear_parameter-varying_control) systems, which a LPTN represents.

