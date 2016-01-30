using Boltzmann
using MNIST

function run_mnist()
	# Set training parameters
	PretrainingEpochs = 2
	Epochs = 3
	Approx = "tap2"
	ApproxSteps = 3
	LearnRate = 0.005
	MonitorEvery = 2
	PersistStart = 5
	Momentum = 0.5
	DecayMagnitude = 0.01
	DecayType = "l2"
	
	# Load MNIST Training data
	X, y = traindata() # Raw data and labels
	normalize_samples!(X) # Pin to range [0,1]
	binarize!(X;threshold=0.001) # Create binary data
	# Hold out a validation set
	TrainSet = X[:,1:1000]
	ValidSet = X[:,59001:end]
	
	# Initialize Model
	HiddenUnits1 = 500
	HiddenUnits2 = 1000d
	rbm1 = BernoulliRBM(28*28, 		HiddenUnits1, (28,28); 			momentum=0., TrainData=TrainSet, sigma = 0.01)
	rbm2 = BernoulliRBM(HiddenUnits1, 	HiddenUnits2, (HiddenUnits1,1); 	momentum=0., sigma = 0.01)
    	layers = [("vishid1",  rbm1),
          		("hid1hid2", rbm2)]
	dbm = DBM(layers)

	# Run pretraining of the model in a greedy layer-wise fashion. This step is optional.
	prefit_dbm = pre_fit(dbm, TrainSet;
				persistent=true, 
				lr=LearnRate, 
				n_iter=PretrainingEpochs, 
				batch_size=100, 
				NormalizationApproxIter=ApproxSteps,
			        	weight_decay="DecayType",decay_magnitude=DecayType,
			        	validation=[],
			        	monitor_every=MonitorEvery,
			        	monitor_vis=true,
			        	approx=Approx,
			        	persistent_start=PersistStart)

	# Run joint training of all the layers of the DBM
	finaldbm,monitor = fit(prefit_dbm, TrainSet; 
				persistent=false, 
				lr=LearnRate, 
				n_iter=Epochs, 
				batch_size=100, 
				NormalizationApproxIter=ApproxSteps,
		             		weight_decay="l2",decay_magnitude=0.01,
		             		validation=ValidSet,
		             		monitor_every=MonitorEvery,
		             		monitor_vis=true,
		             		approx=Approx,
		            		persistent_start=PersistStart)

	# Save plotted charts and raw data monitoring the joint training 
	WriteMonitorChartPDF(finaldbm,monitor,X,"example_dbm_monitor_charts.pdf")
    	SaveMonitorHDF5(monitor,"example_dbm_monitor.h5")

end

run_mnist()
