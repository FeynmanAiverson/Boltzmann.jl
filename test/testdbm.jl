using Boltzmann
using MNIST

function run_mnist()
	X, y = traindata()  
	normalize_samples!(X)
	binarize!(X;threshold=0.01)

	TrainSet = X[:,1:1000]
	ValidSet = X[:,59001:60000]
	Epochs = 1;
	MCMCIter = 1;
	EMFIter = 3
	LearnRate = 0.005
	MonitorEvery = 1
	EMFPersistStart = 5
	HiddenUnits1 = 500
	HiddenUnits2 = 100
	HiddenUnits3 = 10
	approx="naive"

	rbm1 = BernoulliRBM(28*28, 		HiddenUnits1, (28,28); momentum=0., TrainData=TrainSet, sigma = 0.01)
	rbm2 = BernoulliRBM(HiddenUnits1, 	HiddenUnits2, (HiddenUnits1,1); momentum=0., sigma = 0.01)
	rbm3 = BernoulliRBM(HiddenUnits2, 	HiddenUnits3, (HiddenUnits2,1); momentum=0., sigma = 0.01)

    	layers = [("vishid1",  rbm1),
          		("hid1hid2", rbm2),
          		("hid2hid3", rbm3)]
	dbm = DBM(layers)

	prefit_dbm = pre_fit(dbm, TrainSet;
					persistent=false, 
					lr=LearnRate, 
					n_iter=Epochs, 
					batch_size=100, 
					NormalizationApproxIter=EMFIter,
				        	weight_decay="l2",decay_magnitude=0.01,
				        	validation=[],
				        	monitor_every=MonitorEvery,
				        	monitor_vis=true,
				        	approx=approx,
				        	persistent_start=EMFPersistStart)

	finaldbm,monitor = fit(prefit_dbm, TrainSet; 
					persistent=false, 
					lr=LearnRate, 
					n_iter=Epochs, 
					batch_size=100, 
					NormalizationApproxIter=EMFIter,
			             		weight_decay="l2",decay_magnitude=0.01,
			             		validation=ValidSet,
			             		monitor_every=MonitorEvery,
			             		monitor_vis=true,
			             		approx=approx,
			            		persistent_start=EMFPersistStart)
	WriteMonitorChartPDF(finaldbm,monitor,X,"testmonitor_dbm_tap2.pdf")
    	SaveMonitorHDF5(monitor,"testmonitor_dbm_tap2.h5")

end

run_mnist()