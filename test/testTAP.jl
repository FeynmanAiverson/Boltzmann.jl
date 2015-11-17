using Boltzmann
using MNIST
using Base.Test
using HDF5

function run_mnist()
    X, y = traindata()  # test data is smaller, no need to downsample
    normalize_samples!(X)
    binarize!(X;threshold=0.01)

    TrainSet = X
    ValidSet = []
    HiddenUnits = 500;
    Epochs = 20;
    Gibbs = 1;
    IterMag = 3
    LearnRate = 0.005
    MonitorEvery=2
    PersistStart=5

    rbm1 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0, TrainData=TrainSet)
    rbm2 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0, TrainData=TrainSet)
    rbm3 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0, TrainData=TrainSet)


    finalrbmtap2,monitor = fit(rbm1, TrainSet;n_iter=Epochs,
                          weight_decay="l2",
                          decay_magnitude=0.001,
                          lr=LearnRate,
                          persistent=true,
                          validation=ValidSet,
                          n_gibbs=IterMag,
                          monitor_every=MonitorEvery,
                          monitor_vis=true,
                          approx="tap2",
                          persistent_start=PersistStart)

    SaveMonitor(finalrbmtap2,monitor,"testmonitor_tap2.pdf")
    SaveMonitorh5(monitor,"testmonitor_tap2.h5")

    # finalrbmnaive,monitor = fit(rbm2, TrainSet;n_iter=Epochs,
    #                       weight_decay="l2",
    #                       decay_magnitude=0.001,
    #                       lr=LearnRate,
    #                       persistent=true,
    #                       validation=ValidSet,
    #                       n_gibbs=IterMag,
    #                       monitor_every=MonitorEvery,
    #                       monitor_vis=true,
    #                       approx="naive",
    #                       persistent_start=PersistStart)

    # SaveMonitor(finalrbmnaive,monitor,"testmonitor_naive.pdf")
    # SaveMonitorh5(monitor,"testmonitor_naive.h5")

    # finalrbmCD,monitor = fit(rbm3, TrainSet;n_iter=Epochs,
    #                       weight_decay="l2",
    #                       decay_magnitude=0.001,
    #                       lr=LearnRate,
    #                       persistent=true,                          
    #                       validation=ValidSet,
    #                       n_gibbs=Gibbs,
    #                       monitor_every=MonitorEvery,
    #                       monitor_vis=true,
    #                       approx="CD",
    #                       persistent_start=1)

    # SaveMonitor(finalrbmCD,monitor,"testmonitor_CD.pdf")
    # SaveMonitorh5(monitor,"testmonitor_CD.h5")
end

run_mnist()