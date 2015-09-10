
using Boltzmann
using MNIST
using ImageView
using Gadfly

function plot_weights(W, imsize, padding=10)
    h, w = imsize
    n = size(W, 1)
    rows = int(floor(sqrt(n)))
    cols = int(ceil(n / rows))
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
    view(dat)
    return dat
end


function run_mnist()
    X, y = testdata()  # test data is smaller, no need to downsample
    HiddenUnits = 100
    Epochs = 10
    X = X ./ (maximum(X) - minimum(X))
    m = BernoulliRBM(28*28, HiddenUnits) 
    m, historical_pl = fit(m, X; persistent=true, lr=0.1, n_iter=Epochs, batch_size=100, n_gibbs=1, dorate=0.5)
    # plot_weights(m.W[1:64, :], (28, 28))
    plot(x=1:Epochs,y=historical_pl,Geom.line,Guide.ylabel("Pseudo-Liklihood"),Guide.xlabel("Training Epoch"))
    return m
end

run_mnist()

println("Press RETURN when ready")
readline(STDIN)

