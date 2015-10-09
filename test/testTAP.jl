using Colors
using Images
using Boltzmann
using MNIST
using ImageView


function binarize!(x;level=0.001)
  @simd for i=1:length(x)
    @inbounds x[i] = x[i] > level ? 1.0 : 0.0
  end
end


function run_mnist()
    X, y = traindata()  # test data is smaller, no need to downsample
    X = X ./ (maximum(X) - minimum(X))
    binarize!(X)
    HiddenUnits = 500;
    Epochs = 5;
    LearningRate = 0.005;
    

    m1 = BernoulliRBM(28*28, HiddenUnits; momentum=0.0, sigma=0.1)
    m2 = BernoulliRBM(28*28, HiddenUnits; momentum=0.0, sigma=0.1)
    m3 = BernoulliRBM(28*28, HiddenUnits; momentum=0.0, sigma=0.1)
    m4 = BernoulliRBM(28*28, HiddenUnits; momentum=0.0, sigma=0.1)

    mM = BernoulliRBM(28*28, HiddenUnits; momentum=0.0, dataset=X)
    
    info("Running Without Momentum With naive")
    tic()
    rbm1, pseudo1, tap1 = fit(m1, X;n_iter=Epochs,lr=LearningRate, approx="naive",persistent=true, n_gibbs=3, weight_decay="l2",decay_magnitude=-0.001)
    toc()
    chart_weights(m1.W, (28, 28); annotation="naive")
    chart_weights_distribution(m1.W;filename="naive_distribution.pdf",bincount=200)
    chart_likelihood_evolution(pseudo1, tap1; filename="naive_likelihood.pdf")

    info("Running Without Momentum With tap2")
    tic()
    rbm2, pseudo2, tap2 = fit(m2, X; n_iter=Epochs,lr=LearningRate, approx="tap2", persistent=true, n_gibbs=3, weight_decay="l2",decay_magnitude=-0.001)
    toc()
    chart_weights(m2.W, (28, 28); annotation="tap2")
    chart_weights_distribution(m2.W; filename="tap2_distribution.pdf",bincount=200)
    chart_likelihood_evolution(pseudo2, tap2; filename="tap2_likelihood.pdf")

    info("Running Without Momentum With tap3")
    tic()
    rbm3, pseudo3, tap3 = fit(m3, X;n_iter=Epochs,lr=LearningRate, approx="tap3",persistent=true, n_gibbs=3, weight_decay="l2",decay_magnitude=-0.001)
    toc()
    chart_weights(m3.W, (28, 28); annotation="tap3")
    chart_weights_distribution(m3.W;filename="tap3_distribution.pdf",bincount=200)
    chart_likelihood_evolution(pseudo3, tap3; filename="tap3_likelihood.pdf")


    info("Running Without Momentum With CD")
    tic()
    rbm4, pseudo4, tap4 = fit(m4, X;n_iter=Epochs,lr=LearningRate, approx="CD",persistent=true, n_gibbs=1, weight_decay="l2",decay_magnitude=-0.001) #, weight_decay="l2",decay_magnitude=0.05)
    toc()
    chart_weights(m4.W, (28, 28); annotation="CD")
    chart_weights_distribution(m4.W;filename="CD_distribution.pdf",bincount=200)
    chart_likelihood_evolution(pseudo4, tap4; filename="CD_likelihood.pdf")
    # return m
end

run_mnist()

println("Press RETURN when ready")
readline(STDIN)

