using Boltzmann
using Base.Test

usingApple = @osx? true : false

if usingApple
    ### 1. Test the bottleneck logsig function
    L = 256
    N = L^2
    Trials = 25
    Xvec = 2*randn(N)
    Xmat = 2*randn(L,L)


    Boltzmann.logistic(Xmat)
    Boltzmann.logisticAccel(Xvec)
    Boltzmann.logisticAccel(Xmat)

    # Standard sigmoid operation
    println("Testing AppleAccelerate Logsigmoid ($Trials Runs)")
    println("===================================================")
    # No Accel, Vector
        Boltzmann.logistic(Xvec)        # Warmstart
        tic(); 
        for itr=1:Trials
            Boltzmann.logistic(Xvec)
        end
        tV=toq()
    println("   [NoAccel] Vector ($N x 1): $tV sec")

    # No Accel, Matrix
        Boltzmann.logistic(Xmat)        # Warmstart
        tic(); 
        for itr=1:Trials
            Boltzmann.logistic(Xmat)
        end
        tM=toq()
    println("   [NoAccel] Matirx ($L x $L): $tM sec")

    # Accel, Vector
        Boltzmann.logisticAccel(Xvec)      # Warmstart
        tic(); 
        for itr=1:Trials
            Boltzmann.logisticAccel(Xvec)
        end
        tVAccel=toq()
    println("   [Accel]   Vector ($N x 1): $tVAccel sec")

    # Accel, Matrix
        Boltzmann.logisticAccel(Xmat)        # Warmstart
        tic(); 
        for itr=1:Trials
            Boltzmann.logisticAccel(Xmat)
        end
        tMAccel=toq()
    println("   [Accel]   Matirx ($L x $L): $tMAccel sec")

    vecBoost = tV./tVAccel
    matBoost = tM./tMAccel
    mse = mean((Boltzmann.logistic(Xvec) - Boltzmann.logisticAccel(Xvec)).^2)
    println("===================================================")
    println("Vector Accel Calculation: $vecBoost Times Faster")
    println("Matrix Accel Calculation: $matBoost Times Faster")
    println("Discrepency: $mse (MSE)")
    println("===================================================")

    ### 2. Test the Accelerated version of fit()
    NFeatures = 784
    NHidden = 300
    NTrain = 10000
    Epochs = 2

    X = rand(NFeatures,NTrain)
    rbm = BernoulliRBM(NFeatures,NHidden)

    println("")
    println("RBM Tests")
    println("===================================================")    
    info("WARMUP")
    fit(rbm,X;n_iter=Epochs)     # Warmup
    info("No Accel")
    tic()
        fit(rbm,X;n_iter=Epochs)
    tRBM = toq()
    info("==> Walltime: $tRBM sec")
    println("")

    info("WARMUP")
    fit(rbm,X;accelerate=true,n_iter=Epochs)      # Warmup
    info("With Accel")
    tic()
        fit(rbm,X;accelerate=true,n_iter=Epochs)
    tRBMAccel = toq()
    info("==> Walltime: $tRBMAccel sec")

    rbmBoost = tRBM ./ tRBMAccel

    println("===================================================")
    println("$Epochs Iteration Accel PCD: $rbmBoost Times Faster")
    println("===================================================")
end