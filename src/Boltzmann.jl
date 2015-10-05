
module Boltzmann

export RBM,
       BernoulliRBM,
       GRBM,
       DBN,
       DAE,
       fit,
       transform,
       generate,
       components,
       features,
       unroll,
       save_params,
       load_params,
       chart_weights,
       chart_weights_distribution,
       chart_activation_distribution,
       binarize,
       binarize!,
       normalize,
       normalize!,
       normalize_samples,
       normalize_samples!,
       removemean,
       removemean!

include("core.jl")

end
