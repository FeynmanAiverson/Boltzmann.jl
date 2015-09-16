using Colors
using Images
using Boltzmann
using MNIST
using ImageView

function normalize(x)
    x=(x-minimum(x)) ./ (maximum(x) - minimum(x))
    return x
end

function plot_weights(W, imsize; padding=0, annotation="")
    h, w = imsize
    n = size(W, 1)
    rows = round(Int,floor(sqrt(n)))
    cols = round(Int,ceil(n / rows))
    halfpad = div(padding, 2)
    dat = zeros(rows * (h + padding), cols * (w + padding))
    for i=1:n
        wt = W[i, :]
        wim = reshape(wt, imsize)
        wim = wim ./ (maximum(wim) - minimum(wim))
        r = div(i - 1, cols) + 1
        c = rem(i - 1, cols) + 1
        dat[(r-1)*(h+padding)+halfpad+1 : r*(h+padding)-halfpad,
            (c-1)*(w+padding)+halfpad+1 : c*(w+padding)-halfpad] = wim
    end

    dat=normalize(dat)
    imgc,img = ImageView.view(dat)
    ImageView.annotate!(imgc,img,ImageView.AnnotationText(20,20,annotation,color=RGB(1,1,1),fontsize=14,halign="left"))

    return dat
end


function run_mnist()
    X, y = testdata()  # test data is smaller, no need to downsample
    X = X ./ (maximum(X) - minimum(X))
    X = X[:,1:1000]
    HiddenUnits = 100;

    m = BernoulliRBM(28*28, HiddenUnits)
    mwdQuad = BernoulliRBM(28*28, HiddenUnits;momentum=0.2)
    mwdLin = BernoulliRBM(28*28, HiddenUnits;momentum=0.2)
    
    # Attempt without weight decay
    info("Running Without Weight Decay")
    fit(m, X;n_iter=100)
    plot_weights(m.W, (28, 28); annotation="No Weight Decay")

    # Attempt with L2 weight decay
    info("Running With L2-Decay")
    fit(mwdQuad, X;n_iter=100,weight_decay="l2",decay_magnitude=0.1,lr=0.001)
    plot_weights(mwdQuad.W, (28, 28);annotation="L2 Weight Decay")

    # Attempt with L1 weight decay
    info("Running With L1-Decay")
    fit(mwdLin, X;n_iter=100,weight_decay="l1",decay_magnitude=0.1,lr=0.001)
    plot_weights(mwdLin.W, (28, 28);annotation="L1 Weight Decay")

    return m
end

run_mnist()

println("Press RETURN when ready")
readline(STDIN)

