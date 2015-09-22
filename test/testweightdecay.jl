using Colors
using Images
using Boltzmann
using MNIST
using ImageView
using Base.Test

function run_mnist()
    X, y = testdata()  # test data is smaller, no need to downsample
    X = X ./ (maximum(X) - minimum(X))

    TrainSet = X[:,1:7000]
    ValidSet = X[:,7001:end]
    HiddenUnits = 500;
    Epochs = 10;

    m = BernoulliRBM(28*28, HiddenUnits; momentum=0.5, dataset=TrainSet)
    mwdQuad = BernoulliRBM(28*28, HiddenUnits; momentum=0.5 ,dataset=TrainSet)
    mwdLin = BernoulliRBM(28*28, HiddenUnits; momentum=0.5, dataset=TrainSet)
    
    # Attempt without weight decay
    info("Running Without Weight Decay")
    fit(m, TrainSet;n_iter=Epochs,lr=0.005,validation=ValidSet)
    chart_weights(m.W, (28, 28); annotation="No Weight Decay")
    chart_weights_distribution(m.W;filename="nodecay_distribution.pdf",bincount=200)

    # Attempt with L2 weight decay
    info("Running With L2-Decay")
    fit(mwdQuad, TrainSet;n_iter=Epochs,weight_decay="l2",decay_magnitude=0.05,lr=0.005,validation=ValidSet)
    chart_weights(mwdQuad.W, (28, 28);annotation="L2 Weight Decay")
    chart_weights_distribution(mwdQuad.W;filename="l2decay_distribution.pdf",bincount=200)

    # Attempt with L1 weight decay
    info("Running With L1-Decay")
    fit(mwdLin, TrainSet;n_iter=Epochs,weight_decay="l1",decay_magnitude=0.05,lr=0.005,validation=ValidSet)
    chart_weights(mwdLin.W, (28, 28);annotation="L1 Weight Decay")
    chart_weights_distribution(mwdLin.W;filename="l1decay_distribution.pdf",bincount=200)

    return m
end

run_mnist()


