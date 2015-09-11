# This file is a demo of the Boltzmann package on mnist

## Use if needed
## Pkg.update()
## Pkg.clone("https://github.com/sphinxteam/Boltzmann.jl")

pwd()
cd("/home/aurelien/work/Boltzmann/demo")

using Boltzmann
using MNIST
using PyPlot
using Images

# load mnist data (samples and labels)
trainX,trainY = traindata()

# extract the training set
nb_training_samples = 10000
X = trainX[:,1:nb_training_samples]
Y = trainY[1:nb_training_samples]

Xtest = trainX[:,nb_training_samples+1:2*nb_training_samples]
Ytest = trainY[nb_training_samples+1:2*nb_training_samples]

# rescale the image in [0:1]
X = (X - minimum(X)) / (maximum(X) - minimum(X))

nb_vis = size(X[:,1])[1]
nb_hid = 100
dbn_layers = [("vis", BernoulliRBM(nb_vis, nb_hid))]

# Case for a rbm
rbm = BernoulliRBM(nb_vis,nb_hid)

# Training the network
dbn = DBN(dbn_layers)
fit(dbn, X, n_iter=50)
fit(rbm, X, n_iter=30)
## Pkg.add("Images")

## Pkg.add("PyPlot")



# Saving hundred features of the network
for i in 1:10
  for j in 1:10
    ind=i+(j-1)*10
    REW = (rbm.W[ind,:] - minimum(rbm.W[ind,:])) / (maximum(rbm.W[ind,:]) - minimum(rbm.W[ind,:]))
    subplot(10,10,ind)
    imshow(reshape(REW,28,28),cmap="gray")
    ax = gca() # see https://gist.github.com/gizmaa/7214002
    setp(ax[:get_xticklabels](),visible=false) # hide xtick
    setp(ax[:get_yticklabels](),visible=false) # hide ytick
    ax[:xaxis][:set_tick_params](which="major",length=0,width=0)
    ax[:yaxis][:set_tick_params](which="major",length=0,width=0)
  end
end
savefig("test.png")


# For multiclass prediction, one can use Mocha softmax layer
##Pkg.add("Mocha")
using Mocha
using HDF5
const RBM_PATH = "/tmp/rbm_mnist.hdf5"
save_params(RBM_PATH, dbn)




backend = CPUBackend()
data_layer = MemoryDataLayer(name="train-data",tops=[:data,:label], batch_size=500, data=Array[X,Y])
vis_layer = InnerProductLayer(name="vis", output_dim=100, tops=[:vis], bottoms=[:data])
loss_layer = SoftmaxLossLayer(name="loss", bottoms=[:vis, :label])

net = Net("MNIST-TEST", backend, [data_layer,vis_layer,loss_layer])

# finally, load pretrained parameters into corresponding layers of Mocha network
h5open(RBM_PATH) do h5
    load_network(h5, net)
    # if architecture of Mocha network doesn't match DBN exactly, one can
    # additonally pass `die_if_not_found=false` to ignore missing layers and
    # use default initializers instead
    #  load_network(h5, net, die_if_not_found=false)
end

exp_dir = "snapshots"
method = SGD()

params = make_solver_parameters(method, max_iter=10000, regu_coef=0.0005,
                                mom_policy=MomPolicy.Fixed(0.9),
                                lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
                                load_from=exp_dir)
solver = Solver(method, params)


setup_coffee_lounge(solver, save_into="$exp_dir/statistics_.jld", every_n_iter=1000)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

# show performance on test data every 1000 iterations
## data_layer_test = HDF5DataLayer(name="test-data", source="data/test.txt", batch_size=100)
data_layer_test = MemoryDataLayer(name="test-data", data=Array[Xtest,Ytest], batch_size=100)
acc_layer = AccuracyLayer(name="test-accuracy", bottoms=[:vis, :label])
test_net = Net("MNIST-test", backend, [data_layer_test, vis_layer, acc_layer])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

solve(solver, net)

