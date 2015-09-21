
using Boltzmann
using MNIST
using ImageView
using Gadfly
using DataFrames

function run_mnist()
    # Configure Test
    X, y = testdata()  
    HiddenUnits = 256
    Epochs = 15    
    X = X ./ (maximum(X) - minimum(X))
    m_do = BernoulliRBM(28*28, HiddenUnits; momentum=0.95)     
    m = BernoulliRBM(28*28, HiddenUnits; momentum = 0.5)     

    # Fit Models
    m_do, historical_pl_do = fit(m_do, X; persistent=false, lr=0.1, n_iter=Epochs, batch_size=100, 
                                          n_gibbs=1, dorate=0.5, weight_decay="l1",decay_magnitude=0.1)
    m, historical_pl = fit(m, X; persistent=true, lr=0.1, n_iter=Epochs, batch_size=100, 
                                 n_gibbs=1, dorate=0.0, weight_decay="l1",decay_magnitude=0.1)

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

