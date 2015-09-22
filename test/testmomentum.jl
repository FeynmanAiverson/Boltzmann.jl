using Colors
using Images
using Boltzmann
using MNIST
using ImageView
using Base.Test

function run_mnist()
    X, y = testdata()  # test data is smaller, no need to downsample
    binarize!(X)

    HiddenUnits = 256;
    Epochs = 10;
    LearningRate = 0.1;

    m = BernoulliRBM(28*28, HiddenUnits; momentum=0.0, dataset=X)
    mM = BernoulliRBM(28*28, HiddenUnits; momentum=0.5, dataset=X)
    
    # Attempt without Momentum
    info("Running Without Momentum")
    fit(m, X;n_iter=Epochs,lr=LearningRate)
    chart_weights(m.W, (28, 28); annotation="No Momentum")
    chart_weights_distribution(m.W;filename="nomomentum_distribution.pdf",bincount=200)

    # Attempt with Momentum
    info("Running With Momentum")
    fit(mM, X;n_iter=Epochs,lr=LearningRate)
    chart_weights(mM.W, (28, 28); annotation="With Momentum")
    chart_weights_distribution(mM.W;filename="momentum_distribution.pdf",bincount=200)

    return m
end

run_mnist()