using Boltzmann
using MNIST
using Base.Test

function run_mnist()
    X, y = traindata()  # test data is smaller, no need to downsample
    normalize_samples!(X)
    binarize!(X;level=0.2)



    # TrainSet = X
    # ValidSet = []
    TrainSet = X[:,1:9000]
    ValidSet = X[:,59001:end]
    HiddenUnits = 500;
    Epochs = 2;
    Gibbs = 1;
    IterMag = 3
    LearnRate = 0.01
    MonitorEvery=1
    PersistStart=5

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
                          approx="tap2",
                          persistent_start=PersistStart)

    SaveMonitor(finalrbm,monitor,"testmonitor_tap2.pdf")
    SaveMonitorh5(monitor,"testmonitor_tap2.h5")

    # finalrbm,monitor = fit(rbm2, TrainSet;n_iter=Epochs,
    #                       weight_decay="l2",
    #                       decay_magnitude=0.001,
    #                       lr=LearnRate,
    #                       validation=ValidSet,
    #                       n_gibbs=IterMag,
    #                       monitor_every=MonitorEvery,
    #                       monitor_vis=true,
    #                       approx="naive",
    #                       persistent_start=PersistStart)

    # SaveMonitor(finalrbm,monitor,"testmonitor_naive.pdf")

    # finalrbm,monitor = fit(rbm3, TrainSet;n_iter=Epochs,
    #                       weight_decay="l2",
    #                       decay_magnitude=0.001,
    #                       lr=LearnRate,
    #                       validation=ValidSet,
    #                       n_gibbs=Gibbs,
    #                       monitor_every=MonitorEvery,
    #                       monitor_vis=true,
    #                       approx="CD",
    #                       persistent_start=PersistStart)

    # SaveMonitor(finalrbm,monitor,"testmonitor_CD.pdf")


end

run_mnist()