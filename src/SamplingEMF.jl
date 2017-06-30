using Distributions
using Base.LinAlg.BLAS

### Base MF definitions
#### Naive mean field
function mag_vis_naive(rbm::RBM, m_hid::Mat{Float64}) ## to be constrained to being only Bernoulli
    buf = gemm('T', 'N', rbm.W, m_hid) .+ rbm.vbias
    return logsig(buf)
end    
# Defining a method with same arguments as other mean field approxiamtions
mag_vis_naive(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})=mag_vis_naive(rbm, m_hid) 

function mag_hid_naive(rbm::RBM, m_vis::Mat{Float64}) 
    buf = gemm('N', 'N', rbm.W, m_vis) .+ rbm.hbias
    return logsig(buf)
end    

mag_hid_naive(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})=mag_hid_naive(rbm, m_vis) 

## DBM : Defining a method for  intermediate hidden layers with top down feedback ##
function mag_hid_naive(rbm1::RBM, m_vis::Mat{Float64}, rbm2::RBM, m_hid2::Mat{Float64}) 
    buf = rbm1.hbias .+ gemm('N', 'N', rbm1.W, m_vis) .+ gemm('T', 'N', rbm2.W, m_hid2)   
    return logsig(buf)
end

mag_hid_naive(rbm1::RBM, m_vis::Mat{Float64}, rbm2::RBM, 
                                m_hid2::Mat{Float64}, m_hid1::Mat{Float64}) = mag_hid_naive(rbm1, m_vis, rbm2, m_hid2)

## DBM : Defining a method for first hidden layer considering clamped visible units ##
function mag_hid_clamped_naive(rbm1::RBM, vis::Mat{Float64}, rbm2::RBM, m_hid2::Mat{Float64})
    buf = rbm1.hbias .+ gemm('N', 'N', rbm1.W, vis) .+ gemm('T', 'N', rbm2.W, m_hid2)   
    return logsig(buf)
end

mag_hid_clamped_naive(rbm1::RBM, vis::Mat{Float64}, rbm2::RBM, m_hid2::Mat{Float64}, m_hid1::Mat{Float64}) = mag_hid_clamped_naive(rbm1, vis, rbm2, m_hid2)
  
#### Second order development
function mag_vis_tap2(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64}) ## to be constrained to being only Bernoulli
    buf = gemm('T', 'N', rbm.W, m_hid) .+ rbm.vbias
    second_order = gemm('T', 'N', rbm.W2, m_hid-abs2(m_hid)).*(0.5-m_vis)
    axpy!(1.0, second_order, buf)
    return logsig(buf)
end  

function mag_hid_tap2(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})
    buf = gemm('N', 'N', rbm.W, m_vis) .+ rbm.hbias
    second_order = gemm('N', 'N', rbm.W2, m_vis-abs2(m_vis)).*(0.5-m_hid)
    axpy!(1.0, second_order, buf)
    return logsig(buf)
end

## DBM : Defining a method for  intermediate hidden layers with top down feedback ##
function mag_hid_tap2(rbm1::RBM, m_vis::Mat{Float64}, rbm2::RBM, m_hid2::Mat{Float64}, m_hid1::Mat{Float64})
    buf = rbm1.hbias .+ gemm('N', 'N', rbm1.W, m_vis) .+ gemm('T', 'N', rbm2.W, m_hid2)   
    second_order = gemm('N', 'N', rbm1.W2, m_vis-abs2(m_vis)).*(0.5-m_hid1) .+ gemm('T', 'N', rbm2.W2, m_hid2-abs2(m_hid2)).*(0.5-m_hid1)
    axpy!(1.0, second_order, buf)
    return logsig(buf)
end

## DBM : Defining a method for first hidden layer considering clamped visible units ##
function mag_hid_clamped_tap2(rbm1::RBM, vis::Mat{Float64}, rbm2::RBM, m_hid2::Mat{Float64}, m_hid1::Mat{Float64})
    buf = rbm1.hbias .+ gemm('N', 'N', rbm1.W, vis) .+ gemm('T', 'N', rbm2.W, m_hid2)   
    second_order = gemm('T', 'N', rbm2.W2, m_hid2-abs2(m_hid2)).*(0.5-m_hid1)
    axpy!(1.0, second_order, buf)
    return logsig(buf)
end

#### Third order development
function mag_vis_tap3(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64}) ## to be constrained to being only Bernoulli
    buf = gemm('T', 'N', rbm.W, m_hid) .+ rbm.vbias
    second_order = gemm('T', 'N', rbm.W2, m_hid-abs2(m_hid)).*(0.5-m_vis)
    third_order = gemm('T', 'N', rbm.W3, abs2(m_hid).*(1.-m_hid)).*(1/3-2*(m_vis-abs2(m_vis)))
    axpy!(1.0, second_order, buf)
    axpy!(1.0, third_order, buf)
    return logsig(buf)
end  

function mag_hid_tap3(rbm::RBM, m_vis::Mat{Float64}, m_hid::Mat{Float64})
    buf = gemm('N', 'N', rbm.W, m_vis) .+ rbm.hbias
    second_order = gemm('N', 'N', rbm.W2, m_vis-abs2(m_vis)).*(0.5-m_hid)
    third_order = gemm('N', 'N', rbm.W3, abs2(m_vis).*(1.-m_vis)).*(1/3-2*(m_hid-abs2(m_hid)))
    axpy!(1.0, second_order, buf)
    axpy!(1.0, third_order, buf)
    return logsig(buf)
end

## DBM : Defining a method for  intermediate hidden layers with top down feedback ##
function mag_hid_tap3(rbm1::RBM, m_vis::Mat{Float64}, rbm2::RBM, m_hid2::Mat{Float64}, m_hid1::Mat{Float64})
    buf = rbm1.hbias .+ gemm('N', 'N', rbm1.W, m_vis) .+ gemm('T', 'N', rbm2.W, m_hid2)   
    second_order = gemm('N', 'N', rbm1.W2, m_vis-abs2(m_vis)).*(0.5-m_hid1) .+ gemm('T', 'N', rbm2.W2, m_hid2-abs2(m_hid2)).*(0.5-m_hid1)
    third_order = gemm('N', 'N', rbm1.W3, abs2(m_vis).*(1.-m_vis)).*(1/3-2*(m_hid1-abs2(m_hid1))) .+ gemm('T', 'N', rbm2.W3, abs2(m_hid2).*(1.-m_hid2)).*(1/3-2*(m_hid1-abs2(m_hid1)))
    axpy!(1.0, second_order, buf)
    axpy!(1.0, third_order, buf)
    return logsig(buf)
end

## DBM : Defining a method for first hidden layer considering clamped visible units ##
function mag_hid_clamped_tap3(rbm1::RBM, vis::Mat{Float64}, rbm2::RBM, m_hid2::Mat{Float64}, m_hid1::Mat{Float64})
    buf = rbm1.hbias .+ gemm('N', 'N', rbm1.W, vis) .+ gemm('T', 'N', rbm2.W, m_hid2)   
    second_order = gemm('T', 'N', rbm2.W2, m_hid2-abs2(m_hid2)).*(0.5-m_hid1)
    third_order = gemm('T', 'N', rbm2.W3, abs2(m_hid2).*(1.-m_hid2)).*(1/3-2*(m_hid1-abs2(m_hid1)))
    axpy!(1.0, second_order, buf)
    axpy!(1.0, third_order, buf)
    return logsig(buf)
end

##########################################################################
function equilibrate(rbm::RBM, vis_init::Mat{Float64}, hid_init::Mat{Float64}; iterations=3, approx="tap2", damp=0.5)
    # Redefine names for clarity
    m_vis = copy(vis_init)
    m_hid = copy(hid_init)
    
    # Set the proper iteration based on the approximation type
    if approx == "naive"
        mag_vis = mag_vis_naive
        mag_hid = mag_hid_naive
    elseif approx == "tap3"
        mag_vis = mag_vis_tap3
        mag_hid = mag_hid_tap3
    else    
        mag_vis = mag_vis_tap2
        mag_hid = mag_hid_tap2
    end 
    # Take the desired number of steps
    for i=1:iterations
       m_vis = 0.5 * mag_vis(rbm, m_vis, m_hid) + 0.5 * m_vis
       m_hid = 0.5 * mag_hid(rbm, m_vis, m_hid) + 0.5 * m_hid
    end

    return m_vis, m_hid
end

### Defining DBM methods 
function equilibrate(dbm::DBM, vis_init::Mat{Float64}, array_hid_init::Array{Array{Float64},1}; iterations=3, approx="tap2", damp=0.5)
   # Redefine names for clarity
   m_vis = copy(vis_init)
   array_m_hid = copy(array_hid_init)
   depth = length(array_hid_init)

   # Set the proper iteration based on the approximation type
   if approx == "naive"
      mag_vis = mag_vis_naive    
      mag_hid = mag_hid_naive
   elseif approx == "tap3"
      mag_vis = mag_vis_tap3  
      mag_hid = mag_hid_tap3
   else    
      mag_vis = mag_vis_tap2
      mag_hid = mag_hid_tap2
   end 

   # Take the desired number of steps
   for i=1:iterations
      # update vis and even hidden layers
      m_vis = damp * mag_vis(dbm[1], m_vis, array_m_hid[1]) + (1-damp) * m_vis
      for k=2:2:depth-1
         array_m_hid[k] = damp * mag_hid(dbm[k],array_m_hid[k-1],dbm[k+1],array_m_hid[k+1],array_m_hid[k]) + (1-damp) * array_m_hid[k]
      end
      if iseven(depth)
         k=depth
         array_m_hid[k] = damp * mag_hid(dbm[k],array_m_hid[k-1],array_m_hid[k]) + (1-damp) * array_m_hid[k]
      end

      # update odd hidden layers 
      array_m_hid[1] = damp * mag_hid(dbm[1], m_vis, dbm[2], array_m_hid[2], array_m_hid[1]) + (1-damp) * array_m_hid[1]
      for k=3:2:depth-1
         array_m_hid[k] = damp * mag_hid(dbm[k],array_m_hid[k-1],dbm[k+1],array_m_hid[k+1],array_m_hid[k]) + (1-damp) * array_m_hid[k]
      end
      if isodd(depth)
         k=depth
         array_m_hid[k] = damp * mag_hid(dbm[k],array_m_hid[k-1],array_m_hid[k]) + (1-damp) * array_m_hid[k]
      end
   end

   return m_vis, array_m_hid
end

function clamped_equilibrate(dbm::DBM, vis::Mat{Float64}, array_hid_init::Array{Array{Float64},1}; iterations=3, approx="tap2", damp=0.5)
   # Redefine names for clarity
   array_m_hid = copy(array_hid_init)
   depth = length(array_hid_init)

   # Set the proper iteration based on the approximation type
   if approx == "naive" 
      mag_hid = mag_hid_naive
      mag_hid_clamped = mag_hid_clamped_naive
   elseif approx == "tap3"  
      mag_hid = mag_hid_tap3
      mag_hid_clamped = mag_hid_clamped_tap3
   else    
      mag_hid = mag_hid_tap2
      mag_hid_clamped = mag_hid_clamped_tap2
   end 

   # Take the desired number of steps
   for i=1:iterations
      # update vis and even hidden layers
      for k=2:2:depth-1
         array_m_hid[k] = damp * mag_hid(dbm[k],array_m_hid[k-1],dbm[k+1],array_m_hid[k+1],array_m_hid[k]) + (1-damp) * array_m_hid[k]
      end
      if iseven(depth)
         k=depth
         array_m_hid[k] = damp * mag_hid(dbm[k],array_m_hid[k-1],array_m_hid[k]) + (1-damp) * array_m_hid[k]
      end

      # update odd hidden layers 
      array_m_hid[1] = damp * mag_hid_clamped(dbm[1], vis, dbm[2], array_m_hid[2], array_m_hid[1]) + (1-damp) * array_m_hid[1]
      for k=3:2:depth-1
         array_m_hid[k] = damp * mag_hid(dbm[k],array_m_hid[k-1],dbm[k+1],array_m_hid[k+1],array_m_hid[k]) + (1-damp) * array_m_hid[k]
      end
      if isodd(depth)
         k=depth
         array_m_hid[k] = damp * mag_hid(dbm[k],array_m_hid[k-1],array_m_hid[k]) + (1-damp) * array_m_hid[k]
      end
   end

   return vis, array_m_hid
end


##### For adatap
function reg_logsig(x::Mat{Float64};thres=1e-5)
    return max(min(logsig(x),1-thres),thres)
     
end
function reg_logsig(x::Vec{Float64})
    return max(min(logsig(x),1-thres),thres)
     
end

function VAMP(rbm,vis; damp=0.5, tmax=10, b0 = log(0.75/(1-0.75)))

    HiddenUnits,VisibleUnits = size(rbm.W)
    allUnits = VisibleUnits+HiddenUnits
    J = [[zeros(VisibleUnits,VisibleUnits) rbm.W'];[rbm.W zeros(HiddenUnits, HiddenUnits)]]
    h = [rbm.vbias ; rbm.hbias]
    lambda=eigmax(J)
    batchsize = size(vis,2)
    ### this time As is the collection of values on the diagonal (1st dim), for all init in the batch (2nd dim)    


    hid = logsig(rbm.W * vis .+ rbm.hbias) #rand(HiddenUnits).<
    B_1 = [(2*vis-1)*b0; log(hid./(1.-hid))] 
    A_1 = 1e-3*ones(allUnits,batchsize)
    B_2 = zeros(allUnits,batchsize)
    A_2 = zeros(allUnits,batchsize)
    a_2 = zeros(allUnits,batchsize)

    B_1old = zeros(allUnits,batchsize)
    A_1old = zeros(allUnits,batchsize)
    B_2old = zeros(allUnits,batchsize)
    A_2old = zeros(allUnits,batchsize)
    C_2 = zeros(allUnits,allUnits,batchsize)
    a_2old = a_2
    C_2old = C_2

    for t=1:tmax
    #     # damping
        a_1 = reg_logsig(B_1 - 0.5*A_1)                                   #24
        c_1 = a_1 .* (1 - a_1)                                      #24
        A_2 = damp*(1./c_1 - A_1) + (1-damp)*A_2old          #25

        for k=1:batchsize
            if minimum(A_2[:,k]) < lambda 
                println("regularizing A_2: ",minimum(A_2[:,k])-lambda)
                idx = indmin(A_2[:,k])
                multiplier = ceil(c_1[idx,k]/damp*(lambda+damp*A_1[idx,k]+(1-damp)*A_2old[idx,k]))
                A_2[:,k] = damp*((multiplier./c_1[:,k]) - A_1[:,k]) + (1-damp)*A_2old[:,k]
            else 
                multiplier=1
            end
            B_2[:,k] = damp*(multiplier./(1-a_1[:,k]) - B_1[:,k]) + (1-damp)*B_2old[:,k]    #25
            C_2[:,:,k] = inv(diagm(A_2[:,k])-J) #block_inv(A_2,W) #26
            a_2[:,k] = C_2[:,:,k] * (B_2 .+ h)[:,k]
            A_1[:,k] = damp*(1./diag(C_2[:,:,k]) - A_2[:,k] ) + (1-damp)*A_1old[:,k]   #27
            B_1[:,k] = damp*(a_2[:,k]./diag(C_2[:,:,k]) - B_2[:,k]) + (1-damp)*B_1old[:,k]         #27
        end

        diff = mean((a_1 - a_2).^2)
        println("mse(a_1 - a_2):",diff)
        diff < 1e-4 ? break : nothing
        if sum(isnan(a_2))>0 || sum(isnan(C_2))>0 || sum(C_2.==0)>0 || sum(c_1.==0)>0
            C_2 =C_2old
            a_2 =a_2old
            a_1 = logsig(B_1old - 0.5*A_1old)
            diff2 = mean((a_1 - a_2).^2)
            println("correctionmse(a_1 - a_2):",diff2)
            break
        end
        B_1old = B_1
        A_1old = A_1
        B_2old = B_2
        A_2old = A_2

    end
    return a_2, C_2
end

##-----------------------------------------------------------------------------##
### To be deprecated still used in Scoring.jl for the moment
function iter_mag(rbm::RBM, vis::Mat{Float64}; n_times=3, approx="tap2")
    v_pos = vis
    h_pos = ProbHidCondOnVis(rbm, v_pos)

    if approx == "naive"
        mag_vis = mag_vis_naive
        mag_hid = mag_hid_naive
    elseif approx == "tap3"
        mag_vis = mag_vis_tap3
        mag_hid = mag_hid_tap3
    else    
        mag_vis = mag_vis_tap2
        mag_hid = mag_hid_tap2
    end    

    m_vis = 0.5 * mag_vis(rbm, vis, h_pos) + 0.5 * vis
    m_hid = 0.5 * mag_hid(rbm, m_vis, h_pos) + 0.5 * h_pos
    for i=1:n_times-1
       m_vis = 0.5 * mag_vis(rbm, m_vis, m_hid) + 0.5 * m_vis
       m_hid = 0.5 * mag_hid(rbm, m_vis, m_hid) + 0.5 * m_hid
    end

    return v_pos, h_pos, m_vis, m_hid
end    

function iter_mag_persist!(rbm::RBM, vis::Mat{Float64}; n_times=3, approx="tap2")
    v_pos = vis
    h_pos = ProbHidCondOnVis(rbm, v_pos)
    
    if approx == "naive"
        mag_vis = mag_vis_naive
        mag_hid = mag_hid_naive
    elseif approx == "tap3"
        mag_vis = mag_vis_tap3
        mag_hid = mag_hid_tap3
    else    
        mag_vis = mag_vis_tap2
        mag_hid = mag_hid_tap2
    end    

    m_vis = rbm.persistent_chain_vis
    m_hid = rbm.persistent_chain_hid

    for i=1:n_times
       m_vis = 0.5 * mag_vis(rbm, m_vis, m_hid) + 0.5 * m_vis
       m_hid = 0.5 * mag_hid(rbm, m_vis, m_hid) + 0.5 * m_hid
    end

    rbm.persistent_chain_vis = m_vis
    rbm.persistent_chain_hid = m_hid

    return v_pos, h_pos, m_vis, m_hid
end 