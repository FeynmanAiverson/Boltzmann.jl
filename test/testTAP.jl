using Boltzmann
using MNIST
using Base.Test
using HDF5

function run_mnist()
    X, y = traindata()  # test data is smaller, no need to downsample
    normalize_samples!(X)
    binarize!(X;level=0.2)



    # TrainSet = X
    # ValidSet = []
    TrainSet = X[:,1:59000]
    ValidSet = X[:,59001:end]
    HiddenUnits = 500;
    Epochs = 10;
    Gibbs = 1;
    IterMag = 3
    LearnRate = 0.005
    MonitorEvery=1
    PersistStart=5

    rbm1 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0)
    rbm2 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0)
    rbm3 = BernoulliRBM(28*28, HiddenUnits, (28,28); momentum=0.0)


    # Attempt with L2 weight decay
    # finalrbmtap2,monitor = fit(rbm1, TrainSet;n_iter=Epochs,
    #                       weight_decay="l2",
    #                       decay_magnitude=0.001,
    #                       lr=LearnRate,
    #                       persistent=true,
    #                       validation=ValidSet,
    #                       n_gibbs=IterMag,
    #                       monitor_every=MonitorEvery,
    #                       monitor_vis=true,
    #                       approx="tap2",
    #                       persistent_start=PersistStart)

    # SaveMonitor(finalrbmtap2,monitor,"testmonitor_tap2.pdf")
    # SaveMonitorh5(monitor,"testmonitor_tap2.h5")

    finalrbmnaive,monitor = fit(rbm2, TrainSet;n_iter=Epochs,
                          weight_decay="l2",
                          decay_magnitude=0.001,
                          lr=LearnRate,
                          persistent=true,
                          validation=ValidSet,
                          n_gibbs=IterMag,
                          monitor_every=MonitorEvery,
                          monitor_vis=true,
                          approx="naive",
                          persistent_start=PersistStart)

    SaveMonitor(finalrbmnaive,monitor,"testmonitor_naive.pdf")
    SaveMonitorh5(monitor,"testmonitor_naive.h5")

    finalrbmCD,monitor = fit(rbm3, TrainSet;n_iter=Epochs,
                          weight_decay="l2",
                          decay_magnitude=0.001,
                          lr=LearnRate,
                          persistent=true,                          
                          validation=ValidSet,
                          n_gibbs=Gibbs,
                          monitor_every=MonitorEvery,
                          monitor_vis=true,
                          approx="CD",
                          persistent_start=PersistStart)

    SaveMonitor(finalrbmCD,monitor,"testmonitor_CD.pdf")
    SaveMonitorh5(monitor,"testmonitor_CD.h5")


# h5open("finalrbms2.h5","w") do file
#   write(file,"WRBMtap2",finalrbmtap2)
#   write(file,"WRBMtap2",finalrbmtap2.W)
#   write(file,"hbRBMtap2",finalrbmtap2.hbias)
#   write(file,"vbRBMtap2",finalrbmtap2.vbias)
#   write(file,"WRBMCD",finalrbmCD.W)
#   write(file,"hbRBMCD",finalrbmCD.hbias)
#   write(file,"vbRBMCD",finalrbmCD.vbias)
#   write(file,"WRBMnaive",finalrbmnaive.W)
#   write(file,"hbRBMnaive",finalrbmnaive.hbias)
#   write(file,"vbRBMnaive",finalrbmnaive.vbias)
# end

end

run_mnist()




##### Piece of code to compare the TAP likelihood at the end of training

# Epochs = h5open("testmonitor_CD.h5","r") do file 
#           read(file, "Epochs")
#         end

# CD_tl = h5open("testmonitor_CD.h5","r") do file 
#           read(file, "TAPLikelihood")
#         end
# tap2_tl = h5open("testmonitor_tap2.h5","r") do file 
#           read(file, "TAPLikelihood")
#         end

# naive_tl = h5open("testmonitor_CD.h5","r") do file 
#           read(file, "TAPLikelihood")
#         end        

# plt.figure()
# plt.plot(Epochs, CD_tl, "d-", label="CD")
# plt.plot(Epochs, tap2_tl, "^-",label="tap2")
# plt.plot(Epochs, naive_tl, "*-" , label="naive")
# plt.legend()
# plt.show()