
Boltzmann.jl
============

[![Build Status](https://travis-ci.org/dfdx/Boltzmann.jl.svg)](https://travis-ci.org/dfdx/Boltzmann.jl)

Restricted Boltzmann machines and deep belief networks in Julia.
This particular package is a fork of [dfdx/Boltzmann.jl](https://github.com/dfdx/Boltzmann.jl) 
with modificaitons made by the SPHINX Team @ ENS Paris.


Installation
------------
Currently, this package is unregistered with the Julia package manager. Once the modifications
here are feature complete, we can either make the fork permanent or request a merge back into
the [dfdx/Boltzmann.jl](https://github.com/dfdx/Boltzmann.jl) package. For now, installation
should be accomplished via:

```julia
    Pkg.clone("https://github.com/sphinxteam/Boltzmann.jl")
```

RBM Basic Usage
---------------

Below, we show a basic script to train a binary RBM on random training data. For this example, persistent contrastive divergence with one step on the MCMC sampling chain (PCD-1) is used. Finally, we also point out the monitoring and charting functionality which is passed as an optional argument to the `fit` procedure.

```julia
    using Boltzmann

    # Experimental parameters for this smoke test
    NFeatures    = 100
    FeatureShape = (10,10)
    NSamples     = 2000
    NHidden      = 50

    # Generate a random test set in [0,1]
    X = rand(NFeatures, NSamples)    
    binarize!(X;threshold=0.5)                        

    # Initialize the RBM Model
    rbm = BernoulliRBM(NFeatures, NHidden, FeatureShape)

    # Run CD-1 Training with persistence
    rbm = fit(rbm,X; n_iter        = 30,      # Training Epochs
                     batch_size    = 50,      # Samples per minibatch
                     persistent    = true,    # Use persistent chains
                     approx        = "CD",    # Use CD (MCMC) Sampling
                     monitor_every = 1,       # Epochs between scoring
                     monitor_vis   = true)    # Show live charts
```

MNIST Example
-------------

One can find the script for this example inside the `/examples` directory [of the repository](https://github.com/sphinxteam/Boltzmann.jl/blob/master/examples/mnistexample.jl).

Sampling
--------

One can **generate** vectors similar to given ones,

```julia
    x = ... 
    x_new = generate(rbm, x)
```

RBMs can handle both - dense and sparse arrays. It cannot, however, handle DataArrays because it's up to application how to treat missing values.


RBM Variants
------------

Currently, this version of the Boltzmann package only provides support for the following RBM variants:

 - `BernoulliRBM`: RBM with binary visible and hidden units.

Support for real valued visibile units is still in progress. Some basic functionality for this feature was provided in limited, though unverified way, in the [upstream repository of this fork](https://https://github.com/dfdx/Boltzmann.jl). We suggest waiting until a verified implementation of the G-RBM is provided, here.

Integration with Mocha
----------------------

[Mocha.jl](https://github.com/pluskid/Mocha.jl) is an excellent deep learning framework implementing auto-encoders and a number of fine-tuning algorithms. Boltzmann.jl allows to save pretrained model in a Mocha-compatible file format to be used later on for supervised learning. Below is a snippet of the essential API, while complete code is available in [Mocha Export Example](https://github.com/dfdx/Boltzmann.jl/blob/master/examples/mocha_export_example.jl):

```julia
    # pretraining and exporting in Boltzmann.jl
    dbn_layers = [("vis", GRBM(100, 50)),
                  ("hid1", BernoulliRBM(50, 25)),
                  ("hid2", BernoulliRBM(25, 20))]
    dbn = DBN(dbn_layers)
    fit(dbn, X)
    save_params(DBN_PATH, dbn)

    # loading in Mocha.jl
    backend = CPUBackend()
    data = MemoryDataLayer(tops=[:data, :label], batch_size=500, data=Array[X, y])
    vis = InnerProductLayer(name="vis", output_dim=50, tops=[:vis], bottoms=[:data])
    hid1 = InnerProductLayer(name="hid1", output_dim=25, tops=[:hid1], bottoms=[:vis])
    hid2 = InnerProductLayer(name="hid2", output_dim=20, tops=[:hid2], bottoms=[:hid1])
    loss = SoftmaxLossLayer(name="loss",bottoms=[:hid2, :label])
    net = Net("TEST", backend, [data, vis, hid1, hid2])

    h5open(DBN_PATH) do h5
        load_network(h5, net)
    end
```


