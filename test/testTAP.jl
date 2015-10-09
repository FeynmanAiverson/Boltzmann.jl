using Boltzmann
using MNIST
using Base.Test

function run_mnist()
    X, y = testdata()  # test data is smaller, no need to downsample
    normalize_samples!(X)
    binarize!(X;level=0.2)

    TrainSet = X
    ValidSet = []
    # TrainSet = X[:,1:9000]
    # ValidSet = X[:,9001:end]
    HiddenUnits = 500;
    Epochs = 35;
    Gibbs = 3;
    LearnRate = 0.01
    MonitorEvery=5
    PersistStart=5

    rbm = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0,sigma=0.1)

    # Attempt with L2 weight decay
    finalrbm,monitor = fit(rbm, TrainSet;n_iter=Epochs,
                          weight_decay="l2",
                          decay_magnitude=0.001,
                          lr=LearnRate,
                          validation=ValidSet,
                          n_gibbs=Gibbs,
                          monitor_every=MonitorEvery,
                          monitor_vis=true,
                          approx="tap2",
                          persistent_start=PersistStart)

    SaveMonitor(finalrbm,monitor,"testmonitor_tap2.pdf")
end

run_mnist()