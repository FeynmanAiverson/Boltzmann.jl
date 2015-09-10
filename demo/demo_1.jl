# This file is a demo of the Boltzmann package on mnist
Pkg.update()
Pkg.clone("https://github.com/sphinxteam/Boltzmann.jl")
using Boltzmann
using MNIST

# load mnist data (samples and labels)
trainX,trainY = traindata()

# extract the training set
nb_training_samples = 10000
X = trainX[:,1:nb_training_samples]
Y = trainY[1:nb_training_samples]

# rescale the image in [0:1]
X = (X + abs(minimum(X))) / (maximum(X) - minimum(X))

nb_vis = size(X[:,1])[1]
nb_hid = 100
dbn_layers = [("vis", BernoulliRBM(nb_vis, nb_hid))]
# rbm = BernoulliRBM(nb_vis,nb_hid)

# Training the network
dbn = DBN(dbn_layers)
fit(dbn, X)


# For multiclass prediction, one can use Mocha softmax layer
using HDF5
const RBM_PATH = "/tmp/rbm_mnist.hdf5"
save_params(RBM_PATH, dbn)

Pkg.add("Mocha")
Pkg.rm("Mocha")
Pkg.clone("https://github.com/pluskid/Mocha.jl.git")
using Mocha

backend = CPUBackend()
data = MemoryDataLayer(tops=[:data,:label], batch_size=500, data=Array[X,Y])
vis = InnerProductLayer(name="vis", output_dim=100, tops=[:vis], bottoms=[:data])
loss = SoftmaxLossLayer(name="loss", bottoms=[:vis, :label])

net = Net("MNIST-TEST", backend, [data,vis,loss])

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

solve(solver, net)

