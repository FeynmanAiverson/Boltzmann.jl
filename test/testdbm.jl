using Boltzmann
using MNIST

function run_mnist()
    X, y = traindata()  
    normalize_samples!(X)
    binarize!(X;threshold=0.01)

    X=X[:,1:1000]
    TrainSet = X
    ValidSet = []
    Epochs = 10;
    MCMCIter = 1;
    EMFIter = 3
    LearnRate = 0.005
    MonitorEvery=2
    EMFPersistStart=5
    HiddenUnits1 = 500
    HiddenUnits2 = 100
    HiddenUnits3 = 10

    rbm0 = BernoulliRBM(28*28, 			HiddenUnits1, (28,28); momentum=0.5, TrainData=TrainSet, sigma = 0.01)
    rbm1 = BernoulliRBM(HiddenUnits1, 	HiddenUnits2, (HiddenUnits1,1); momentum=0.5, TrainData=TrainSet, sigma = 0.01)
    rbm2 = BernoulliRBM(HiddenUnits2, 	HiddenUnits3, (HiddenUnits2,1); momentum=0.5, TrainData=TrainSet, sigma = 0.01)

	layers = [("vis",  rbm0),
	          ("hid1", rbm1),
	          ("hid2", rbm2)]
	dbm = DBM(layers)

	println(dbm)
	mhid2=ProbHidAtLayerCondOnVis(dbm,X,2)
	println(size(mhid2)) 
	mhid1=ProbHidCondOnNeighbors(dbm[1],X,dbm[2],mhid2)
	println(size(mhid1))
	println(mhid1)  

end

run_mnist()

# fit(dbn, X)
# transform(dbn, X)

# dae = unroll(dbn)
# transform(dae, X)

# save_params("test.hdf5", dbn)
# save_params("test2.hdf5", dae)
# rm("test.hdf5")
# rm("test2.hdf5")

