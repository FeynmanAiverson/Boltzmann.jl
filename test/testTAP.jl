using Colors
using Images
using Boltzmann
using MNIST
using ImageView

function run_mnist()
    X, y = testdata()  # test data is smaller, no need to downsample
    X = X ./ (maximum(X) - minimum(X))
    HiddenUnits = 100;
    Epochs = 5;
    LearningRate = 0.005;
    

    m1 = BernoulliRBM(28*28, HiddenUnits; momentum=0.0, dataset=X)
    m2 = BernoulliRBM(28*28, HiddenUnits; momentum=0.0, dataset=X)
    m3 = BernoulliRBM(28*28, HiddenUnits; momentum=0.0, dataset=X)
    m4 = BernoulliRBM(28*28, HiddenUnits; momentum=0.0, dataset=X)

    mM = BernoulliRBM(28*28, HiddenUnits; momentum=0.0, dataset=X)
    
    # info("Running Without Momentum With naive")
    # tic()
    # fit(m1, X;n_iter=Epochs,lr=LearningRate, approx="naive",persistent=true, n_gibbs=3, weight_decay="l2",decay_magnitude=0.05)
    # toc()
    # chart_weights(m1.W, (28, 28); annotation="naive")
    # chart_weights_distribution(m1.W;filename="naive_distribution.pdf",bincount=200)

    # info("Running Without Momentum With tap2")
    # tic()
    # fit(m2, X;n_iter=Epochs,lr=LearningRate, approx="tap2", persistent=true, n_gibbs=3, weight_decay="l2",decay_magnitude=0.05)
    # toc()
    # chart_weights(m2.W, (28, 28); annotation="tap2")
    # chart_weights_distribution(m2.W;filename="tap2_distribution.pdf",bincount=200)

    # info("Running Without Momentum With tap3")
    # tic()
    # fit(m3, X;n_iter=Epochs,lr=LearningRate, approx="tap3",persistent=true, n_gibbs=3, weight_decay="l2",decay_magnitude=0.05)
    # toc()
    # chart_weights(m3.W, (28, 28); annotation="tap3")
    # chart_weights_distribution(m3.W;filename="tap3_distribution.pdf",bincount=200)


    info("Running Without Momentum With CD")
    tic()
    fit(m4, X;n_iter=Epochs,lr=LearningRate, approx="CD",persistent=true, n_gibbs=1, weight_decay="l2",decay_magnitude=0.05) #, weight_decay="l2",decay_magnitude=0.05)
    toc()
    chart_weights(m4.W, (28, 28); annotation="CD")
    chart_weights_distribution(m4.W;filename="CD_distribution.pdf",bincount=200)

    #return m
end

run_mnist()

println("Press RETURN when ready")
readline(STDIN)

