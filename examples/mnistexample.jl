
using Boltzmann
using MNIST

function run_mnist()
    # Set parameters
    Epochs = 20
    HiddenUnits = 256


    # Get all MNIST training data
    X, labels = traindata()  
    binarize!(X)

    # Split validation set
    TrainSet = X[:,1:50000]
    ValidSet = X[:,50001:end]

    # Initialize Model
    m = BernoulliRBM(28*28, HiddenUnits; momentum=0.5, dataset=TrainSet)

    # Run Training
    fit(m, TrainSet; n_iter=Epochs, 
                     lr=0.005, 
                     weight_decay="l2",
                     decay_magnitude=0.01,
                     persistent=true,
                     validation=ValidSet)

    # Display Result
    chart_weights(m.W,(28,28))
end

run_mnist()

println("Press RETURN when ready")
readline(STDIN)

