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
    HiddenUnits = 50;
    Epochs = 10;
    MCMCIter = 1;
    EMFIter = 3
    EMFPersistStart=5

    # Set Global Training Parameters
    options = Dict()
    options[:epochs] = 10
    options[:batchSize] = 100
    options[:learnRate] = 0.005
    options[:persist] = true
    options[:monitorEvery] = 2
    options[:monitorVis] = true
    options[:weightDecayType] = "l2"
    options[:weightDecayMagnitude] = 0.001
    options[:validationSet] = []
    # CD params
    cdOptions = deepcopy(options)
    cdOptions[:approxType] = "CD"
    # TAP params
    tapOptions = deepcopy(options)
    tapOptions[:approxType] = "tap2"
    tapOptions[:approxIters] = 3
    tapOptions[:persistStart] = 5
    # NMF params
    nmfOptions = deepcopy(tapOptions)
    nmfOptions[:approxType] = "naive"

    # Initialize models
    rbm1 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.5, TrainData=TrainSet, sigma = 0.01)
    rbm2 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.5, TrainData=TrainSet, sigma = 0.01)
    rbm3 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.5, TrainData=TrainSet, sigma = 0.01)

    # Train TAP
    finalrbmtap2,monitor = fit(rbm1, TrainSet, tapOptions)
    write_monitor_chart_pdf(finalrbmtap2,monitor,X,"testmonitor_tap2.pdf")
    save_monitor_hdf5(monitor,"testmonitor_tap2.h5")

    # Train NMF
    finalrbmnaive,monitor = fit(rbm2, TrainSet, nmfOptions)
    write_monitor_chart_pdf(finalrbmnaive,monitor,X,"testmonitor_naive.pdf")
    save_monitor_hdf5(monitor,"testmonitor_naive.h5")

    # Train CD
    finalrbmCD,monitor = fit(rbm3, TrainSet, cdOptions)
    write_monitor_chart_pdf(finalrbmCD,monitor,X,"testmonitor_CD.pdf")
    save_monitor_hdf5(monitor,"testmonitor_CD.h5")
end

run_mnist()