
using Boltzmann
using Distributions
using Base.Test

X = rand(784,300)
rbm = BernoulliRBM(784,100);

function test_chart_weights()
    chart_weights(rbm.W,(28,28);annotation="Test Annotation")
end

function test_chart_weights_distribution()
    chart_weights_distribution(X;bincount=100,filename="distchart.pdf")
end

function test_chart_activation_distribution()
    chart_activation_distribution(rbm,X;bincount=100,filename="activationdistchart.pdf")
end

test_chart_weights()
test_chart_weights_distribution()
test_chart_activation_distribution()

