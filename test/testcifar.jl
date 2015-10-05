# using Colors
# using Images
using Boltzmann
using CIFAR
# using ImageView
using Base.Test

function run_cifar()
    # Experiment Parameters
    testbatch = 1
    ntrain = 5000
    nhidden = 500
    Epochs = 10
    Gibbs = 1
    LearnRate = 0.05
    Momentum = 0.5
    DecayType = "l1"
    DecayMag = 0.1
    Persistent = false
    L = 32                  # CIFAR-10 Images are 32x32

    # Take sames from the first batch
    X,lables,labelnames = traindata(testbatch; normalize_images=true, 
                                               grey=true)  
    TrainSet = X[:,1:ntrain]
    nvis = size(X,1)

    # Intialize RBMs
    rbmraw  = BernoulliRBM(nvis, nhidden; momentum=Momentum)
    rbmreal = GRBM(nvis, nhidden; momentum=Momentum)


    println(rbmraw)
    println(rbmreal)

    # info("Running CIFAR-10 Test")
    # fit(rbmraw, TrainSet; n_iter=Epochs,
    #                       persistent=Persistent,
    #                       weight_decay=DecayType,
    #                       decay_magnitude=DecayMag,
    #                       lr=LearnRate,
    #                       n_gibbs=Gibbs)

    # chart_weights(rbmraw.W, (L, L); annotation="Bernoulli with Normalized [0,1]")

    # # Generate some samples
    # sample = generate(rbmraw, X[:,1:100];n_gibbs=200)
    # chart_weights(sample',(L, L); annotation="Generated Samples")


    return rbmraw
end

run_cifar()

println("Press RETURN when ready")
readline(STDIN)