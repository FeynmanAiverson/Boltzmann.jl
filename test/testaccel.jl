using Boltzmann
using Base.Test

usingApple = @osx? true : false

if usingApple
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
end