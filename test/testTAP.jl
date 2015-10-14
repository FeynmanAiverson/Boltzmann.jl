using Colors
using Images
using Boltzmann
using MNIST
using ImageView
using Base.Test

function run_mnist()
    X, y = traindata()  # test data is smaller, no need to downsample
    normalize_samples!(X)
    binarize!(X)

    TrainSet = X[:,1:59000]
    ValidSet = X[:,59001:end]
    HiddenUnits = 500;
    Epochs = 25;
    Gibbs = 1
    IterMag = 3
    LearnRate = 0.005
    MonitorEvery=1

    rbm1 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0)
    rbm2 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0)
    rbm3 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0)

    # Attempt with L2 weight decay
    finalrbm,monitor = fit(rbm1, TrainSet;n_iter=Epochs,
                          weight_decay="l2",
                          decay_magnitude=0.001,
                          lr=LearnRate,
                          validation=ValidSet,
                          n_gibbs=IterMag,
                          monitor_every=MonitorEvery,
                          monitor_vis=true,
                          approx="tap2")

    SaveMonitor(finalrbm,monitor,"testmonitor_tap2.pdf")

    finalrbm,monitor = fit(rbm2, TrainSet;n_iter=Epochs,
                          weight_decay="l2",
                          decay_magnitude=0.001,
                          lr=LearnRate,
                          validation=ValidSet,
                          n_gibbs=IterMag,
                          monitor_every=MonitorEvery,
                          monitor_vis=true,
                          approx="naive")

    SaveMonitor(finalrbm,monitor,"testmonitor_naive.pdf")

    finalrbm,monitor = fit(rbm3, TrainSet;n_iter=Epochs,
                          weight_decay="l2",
                          decay_magnitude=0.001,
                          lr=LearnRate,
                          validation=ValidSet,
                          n_gibbs=Gibbs,
                          monitor_every=MonitorEvery,
                          monitor_vis=true,
                          approx="CD")

    SaveMonitor(finalrbm,monitor,"testmonitor_CD.pdf")


end

run_mnist()