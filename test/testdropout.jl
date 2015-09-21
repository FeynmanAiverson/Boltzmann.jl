
using Boltzmann
using MNIST
using ImageView
using Gadfly
using DataFrames

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
    # Configure Test
    X, y = testdata()  
    HiddenUnits = 100
    Epochs = 5    
    X = X ./ (maximum(X) - minimum(X))
    m_do = BernoulliRBM(28*28, HiddenUnits)     
    m = BernoulliRBM(28*28, HiddenUnits)     

    # Fit Models
    m_do, historical_pl_do = fit(m_do, X; persistent=true, lr=0.1, n_iter=Epochs, batch_size=100, n_gibbs=1, dorate=0.5)
    m, historical_pl = fit(m, X; persistent=true, lr=0.1, n_iter=Epochs, batch_size=100, n_gibbs=1, dorate=0.0)

    # Put results in dataframe
    NoDropoutActivations = Boltzmann.transform(m,X)
    DropoutActivations = Boltzmann.transform(m_do,X)

    Results = DataFrame(Epochs=[1:Epochs;1:Epochs],PL=[vec(historical_pl_do);vec(historical_pl)],UsingDropout=[trues(Epochs);falses(Epochs)])


    # Plot Pseudo-liklihood
    PLPlot = plot(Results,x="Epochs",y="PL",color="UsingDropout",Geom.line,Guide.ylabel("Pseudo-Liklihood"),Guide.xlabel("Training Epoch"))
    draw(PDF("Dropout_TrainingPL.pdf", 12inch, 9inch), PLPlot)


    # Plot Activations
    Activations = DataFrame(Act=[vec(NoDropoutActivations);vec(DropoutActivations)],UsingDropout=[falses(vec(NoDropoutActivations));trues(vec(DropoutActivations))])
    HAPlot = plot(Activations,x="Act",color="UsingDropout",Geom.histogram(bincount=100,density=true,position=:dodge),Guide.ylabel("Density"),Guide.xlabel("Hidden Layer Activations"))
    draw(PDF("HiddenActivations.pdf", 12inch, 9inch), HAPlot)    

    return m
end

run_mnist()

