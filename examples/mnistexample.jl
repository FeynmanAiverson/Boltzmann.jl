
using Boltzmann
using MNIST

function run_mnist()
    # Set parameters
    Epochs = 20
    HiddenUnits = 256


    # Get all MNIST training data
    X, y = traindata()  # test data is smaller, no need to downsample
    normalize_samples!(X)
    binarize!(X;level=0.2)

    # Split validation set
    Gibbs = 1
    IterMag = 3
    LearnRate = 0.005
    MonitorEvery=5
    PersistStart=5
    TrainSet = X[:,1:50000]
    ValidSet = X[:,50001:end]

    # Initialize Model
    m = BernoulliRBM(28*28, HiddenUnits,(28,28); momentum=0.5, dataset=TrainSet)

    # Run Training
    fit(m, TrainSet; n_iter=Epochs, 
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

    # Display Result
    chart_weights(m.W,(28,28))
end

run_mnist()

println("Press RETURN when ready")
readline(STDIN)

