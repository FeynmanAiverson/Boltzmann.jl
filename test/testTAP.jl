using Colors
using Images
using Boltzmann
using MNIST
using ImageView

function run_mnist()
    X, y = testdata()  # test data is smaller, no need to downsample
    X = X ./ (maximum(X) - minimum(X))
    HiddenUnits = 100;
    Epochs = 1;
    LearningRate = 0.1;
    

    m = BernoulliRBM(28*28, HiddenUnits; momentum=0.0, dataset=X)
    mM = BernoulliRBM(28*28, HiddenUnits; momentum=0.5, dataset=X)
    
    info("Running Without Momentum With naive")
    tic()
    fit(mM, X;n_iter=Epochs,lr=LearningRate, approx="tap3")
    toc()
    chart_weights(mM.W, (28, 28); annotation="naive")
    chart_weights_distribution(mM.W;filename="naive_distribution.pdf",bincount=200)


    info("Running Without Momentum With CD")
    tic()
    fit(m, X;n_iter=Epochs,lr=LearningRate, approx="CD")
    toc()
    chart_weights(m.W, (28, 28); annotation="CD")
    chart_weights_distribution(m.W;filename="CD_distribution.pdf",bincount=200)


    # info("Running Without Momentum With naive")
    # fit(mM, X;n_iter=Epochs, persistent=false, lr=LearningRate, approx=approx)
    # chart_weights(mM.W, (28, 28); annotation="naive non persistent")
    # chart_weights_distribution(mM.W;filename="naive_nonpersistent_distribution.pdf",bincount=200)

    return m
end

run_mnist()

println("Press RETURN when ready")
readline(STDIN)

