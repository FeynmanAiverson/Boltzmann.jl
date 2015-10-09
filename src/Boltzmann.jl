
module Boltzmann

export RBM,
       BernoulliRBM,
       Monitor,
       Update!,
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
       ShowMonitor,
       SaveMonitor,
       plot_scores,
       plot_evolution,
       plot_rf,
       plot_chain,
       plot_vbias,
       plot_weightdist


include("core.jl")

end
