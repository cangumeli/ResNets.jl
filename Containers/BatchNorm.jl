type BatchNorm <: Layer
   range::Range{Int}
   running_mean::Any
   running_var::Any
   momentum::Number
   eps::Number
   function BatchNorm(net::Net, ndims::Int; affine=true, dtype=Array{Float32}, momentum=.9, eps=1e-9)
      running_mean = dtype(zeros(ndims, 1))
      running_var = dtype(ones(ndims, 1))
      if affine
         push!(net, dtype(ones(ndims, 1)))
         push!(net, dtype(zeros(ndims, 1)))
         return new((length(net)-1):length(net), running_mean, running_var, momentum, eps)
      end
      new(length(net):length(net)-1, running_mean, running_var, momentum, eps)
   end
end

function forward(net, bn::BatchNorm, x; mode=:train)
   assert(mode in [:train, :test])
   w = get_params(net, bn)
   affine = length(w) > 0
   #=if affine
      x = auto_convert(w, x, AutoGrad.getval)
   end=#
   if typeof(bn.running_mean) !== typeof(AutoGrad.getval(x))
      bn.running_mean = typeof(x)(bn.running_mean)
   end
   if typeof(bn.running_var) !== typeof(AutoGrad.getval(x))
      bn.running_var = typeof(x)(bn.running_var)
   end
   x_hat = nothing
   if mode == :test
      x_hat = (x .- bn.running_mean) ./ sqrt(bn.running_var .+ bn.eps)
   else
      # Do the computation
      m = size(x, 2)
      mu = sum(x, 2) ./ m
      x_mu = x .- mu
      sigma2 = sumabs2(x_mu, 2) ./ m
      x_hat = x_mu ./ sqrt(sigma2 .+ bn.eps)
      # Update the running stats
      bn.running_mean = bn.momentum * bn.running_mean + (1 - bn.momentum) * AutoGrad.getval(mu)
      bn.running_var = bn.momentum * bn.running_var + (1 - bn.momentum) * AutoGrad.getval(sigma2)
   end
   affine ? (w[1] .* x_hat .+ w[2]) : x_hat
end

# Turn off weight decay
decay_range(layer::BatchNorm) = 1:0
