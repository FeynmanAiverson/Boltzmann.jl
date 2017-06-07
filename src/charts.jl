using PyCall
using PyPlot

function chart_weights(W, imsize; padding=0, annotation="", filename="", noshow=false, ordering=true)
    h, w = imsize
    n = size(W, 1)
    rows = round(Int,floor(sqrt(n)))
    cols = round(Int,ceil(n / rows))
    halfpad = div(padding, 2)
    dat = zeros(rows * (h + padding), cols * (w + padding))

    # Sort receptive fields by energy
    if ordering
        p = sum(W.^2,2)
        p = sortperm(vec(p);rev=true)
        W = W[p,:]
    end

    for i=1:n
        wt = W[i, :]
        wim = reshape(wt, imsize)
        r = div(i - 1, cols) + 1
        c = rem(i - 1, cols) + 1
        dat[(r-1)*(h+padding)+halfpad+1 : r*(h+padding)-halfpad,
            (c-1)*(w+padding)+halfpad+1 : c*(w+padding)-halfpad] = wim
    end

    normalize!(dat)

    return dat
end

function plot_hidden_activations(rbm::RBM,X::Mat{Float64})
    max_samples = 100
    n_samples = min(size(X,2),max_samples)
    x,_ = random_columns(X,n_samples)

    # Get all hidden activations for batch    
    act = ProbHidCondOnVis(rbm,x)
    # Show this matrix of activations
    imshow(act;interpolation="Nearest")
    title("Hidden Unit Activations")
    xlabel("Random Samples")
    ylabel("Hidden Unit Index")
    gray()

end

function plot_scores(mon::Monitor)
    ax_pl = gca()
    ax_re = ax_pl[:twinx]()
    
    hpl = ax_pl[:plot](mon.Epochs,mon.PseudoLikelihood,"b^-",label="Pseudo-Likelihood")
    htl = ax_pl[:plot](mon.Epochs,mon.TAPLikelihood,"g^-",label="Tap-Likelihood")
    if mon.UseValidation
        hvpl = ax_pl[:plot](mon.Epochs,mon.ValidationPseudoLikelihood,"b^:",label="Pseudo-Likelihood (Validation)")
        hvtl = ax_pl[:plot](mon.Epochs,mon.ValidationTAPLikelihood,"g^:",label="Tap-Likelihood (Validation)")
    end
    ax_pl[:set_ylabel]("Normalized Likelihood")
    ax_pl[:set_ylim]((-0.3,0.0))
    
    hre = ax_re[:plot](mon.Epochs,mon.ReconError,"-*r",label="Recon. Error")
    if mon.UseValidation
        hvre = ax_re[:plot](mon.Epochs,mon.ValidationReconError,":*r",label="Recon. Error (Validation)")
    end
    ax_re[:set_ylabel]("Value")            
    ax_re[:set_yscale]("log")

    title("Scoring")
    xlabel("Training Epoch")
    xlim((1,mon.Epochs[mon.LastIndex]))        
    grid("on")        
    if mon.UseValidation
        legend(handles=[hpl;hvpl;htl;hvtl;hre;hvre],loc=2)
    else
        legend(handles=[hpl;htl;hre],loc=2)
    end
end

function plot_evolution(mon::Monitor)
    hbt = plot(mon.Epochs,mon.BatchTime_µs,"-k*",label="Norm. Batch time (µs)")

    legend(handles=hbt,loc=1)
    title("Evolution")
    xlabel("Training Epoch")
    xlim((1,mon.Epochs[mon.LastIndex]))        
    grid("on")  
end

function plot_rf(rbm::RBM)
    # TODO: Implement RF display in the case of 1D signals
    rf = chart_weights(rbm.W,rbm.VisShape; padding=0,noshow=true)    
    imshow(rf;interpolation="Nearest")
    title("Receptive Fields")
    gray()
end

function plot_chain(rbm::RBM)
    # TODO: Implement Chain display in the case of 1D signals
    pc = chart_weights(rbm.persistent_chain_vis',rbm.VisShape; padding=0,noshow=true,ordering=false)    
    imshow(pc;interpolation="Nearest")
    title("Visible Chain")
    gray()
end

function plot_vbias(rbm::RBM)
    vectorMode = minimum(rbm.VisShape)==1 ? true : false

    if vectorMode
        plot(rbm.vbias)
        grid("on")        
    else
        imshow(reshape(rbm.vbias,rbm.VisShape);interpolation="Nearest")
    end
    title("Visible Biasing")
    gray()
end

function plot_weightdist(rbm::RBM)
    hist(vec(rbm.W),100)
    title("Weight Distribution")
    xlabel("Weight Value")
    ylabel("Frequeny")
end

function figure_refresh(figureHandle)
    figureHandle[:canvas][:draw]()
    show()
    pause(0.0001)
end


function WriteMonitorChartPDF(rbm::RBM,mon::Monitor,X::Mat{Float64},filename::AbstractString)
    savefig = figure(5;figsize=(12,15))
    # Show Per-Epoch Progres
    savefig[:add_subplot](321)
        plot_scores(mon)
        
    savefig[:add_subplot](322)
        plot_evolution(mon)      

    # Show Receptive fields
    savefig[:add_subplot](323)
        plot_rf(rbm)

    # Show the Visible chains/fantasy particle
    savefig[:add_subplot](324)
        plot_chain(rbm)

    # Show the current visible biasing
    savefig[:add_subplot](325)
        # plot_vbias(rbm)
        plot_hidden_activations(rbm,X)

    # Show the distribution of weight values
    savefig[:add_subplot](326)
        plot_weightdist(rbm)

    savefig(filename;transparent=true,format="pdf",papertype="a4",frameon=true,dpi=300)
    close()
end




function ShowMonitor(rbm::RBM,mon::Monitor,X::Mat{Float64},itr::Int;filename=[])
    fig = mon.FigureHandle

    if mon.MonitorVisual && itr%mon.MonitorEvery==0
        # Wipe out the figure
        fig[:clf]()

        # Show Per-Epoch Progres
        fig[:add_subplot](321)
            plot_scores(mon)
            
        fig[:add_subplot](322)
            plot_evolution(mon)      

        # Show Receptive fields
        fig[:add_subplot](323)
            plot_rf(rbm)

        # Show the Visible chains/fantasy particle
        fig[:add_subplot](324)
            plot_chain(rbm)

        # Show the current visible biasing
        fig[:add_subplot](325)
            # plot_vbias(rbm)
            plot_hidden_activations(rbm,X)

        # Show the distribution of weight values
        fig[:add_subplot](326)
            plot_weightdist(rbm)

        figure_refresh(fig)  
    end

    if mon.MonitorText && itr%mon.MonitorEvery==0
        li = mon.LastIndex
        ce = mon.Epochs[li]
        if mon.UseValidation
            @printf("[Epoch %04d] Train(pl : %0.3f, tl : %0.3f), Valid(pl : %0.3f, tl : %0.3f)  [%0.3f µsec/batch/unit]\n",ce,
                                                                                                   mon.PseudoLikelihood[li],
                                                                                                   mon.TAPLikelihood[li],
                                                                                                   mon.ValidationPseudoLikelihood[li],
                                                                                                   mon.ValidationTAPLikelihood[li],
                                                                                                   mon.BatchTime_µs[li])
        else
            @printf("[Epoch %04d] Train(pl : %0.3f, tl : %0.3f)  [%0.3f µsec/batch]\n",ce,
                                                                           mon.PseudoLikelihood[li],
                                                                           mon.TAPLikelihood[li],
                                                                           mon.BatchTime_µs[li])
        end
    end
end

##################### DBM methods ###############################################################

function WriteMonitorChartPDF(dbm::DBM,mon::Monitor,X::Mat{Float64},filename::AbstractString)
    tosavefig = figure(5;figsize=(12,15))
    # Show Per-Epoch Progres
    tosavefig[:add_subplot](321)
        plot_scores(mon)
        
    tosavefig[:add_subplot](322)
        plot_evolution(mon)      

    # Show Receptive fields
    tosavefig[:add_subplot](323)
        plot_rf(dbm[1])

    # Show the Visible chains/fantasy particle
    tosavefig[:add_subplot](324)
        plot_chain(dbm[1])

    # Show the current visible biasing
    tosavefig[:add_subplot](325)
        # plot_vbias(rbm)
        plot_hidden_activations(dbm[1],X)

    # Show the distribution of weight values
    tosavefig[:add_subplot](326)
        plot_weightdist(dbm[1])

    savefig(filename;transparent=true,format="pdf",papertype="a4",frameon=true,dpi=300)
    close()
end

function ShowMonitor(dbm::DBM,mon::Monitor,X::Mat{Float64},itr::Int;filename=[])
    fig = mon.FigureHandle

    if mon.MonitorVisual && itr%mon.MonitorEvery==0
        # Wipe out the figure
        fig[:clf]()

        # Show Per-Epoch Progres
        fig[:add_subplot](321)
            plot_scores(mon)
            
        fig[:add_subplot](322)
            plot_evolution(mon)      

        # Show Receptive fields
        fig[:add_subplot](323)
            plot_rf(dbm[1])

        # Show the Visible chains/fantasy particle
        fig[:add_subplot](324)
            plot_chain(dbm[1])

        # Show the current visible biasing
        fig[:add_subplot](325)
            # plot_vbias(rbm)
            plot_hidden_activations(dbm[1],X)

        # Show the distribution of weight values
        fig[:add_subplot](326)
            plot_weightdist(dbm[1])

        figure_refresh(fig)  
    end

    if mon.MonitorText && itr%mon.MonitorEvery==0
        li = mon.LastIndex
        ce = mon.Epochs[li]
        if mon.UseValidation
            @printf("[Epoch %04d] Train(pl : %0.3f, tl : %0.3f), Valid(pl : %0.3f, tl : %0.3f)  [%0.3f µsec/batch/unit]\n",ce,
                                                                                                   mon.PseudoLikelihood[li],
                                                                                                   mon.TAPLikelihood[li],
                                                                                                   mon.ValidationPseudoLikelihood[li],
                                                                                                   mon.ValidationTAPLikelihood[li],
                                                                                                   mon.BatchTime_µs[li])
        else
            @printf("[Epoch %04d] Train(pl : %0.3f, tl : %0.3f)  [%0.3f µsec/batch]\n",ce,
                                                                           mon.PseudoLikelihood[li],
                                                                           mon.TAPLikelihood[li],
                                                                           mon.BatchTime_µs[li])
        end
    end
end
# function chart_likelihood_evolution(pseudo, tap; filename="")

#     if length(filename) > 0
#         # Write to file if filename specified
#         LikelihoodPlot = plot(  
#                                 layer(x=1:length(pseudo),y=pseudo,Geom.point, Geom.line),
#                                 layer(x=1:length(tap),y=tap, Geom.point, Geom.line, Theme(default_color=colorant"green")),
#                                 Guide.xlabel("epochs"),Guide.ylabel("Likelihood"),Guide.title("Evolution of likelihood for training set")
#                             )
#         draw(PDF(filename, 10inch, 6inch), LikelihoodPlot)
#     else
#         # Draw plot if no filename given
#        plot(  
#                                 layer(x=1:length(pseudo),y=pseudo,Geom.point, Geom.line),
#                                 layer(x=1:length(tap),y=tap, Geom.point, Geom.line, Theme(default_color=colorant"green")),
#                                 Guide.xlabel("epochs"),Guide.ylabel("Likelihood"),Guide.title("Evolution of likelihood for training set")
#                             )
#     end
# end