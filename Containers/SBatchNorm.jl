type SBatchNorm <: Layer
   range::Range{Int}
   running_mean::Any
   running_var::Any
   momentum::Number
   eps::Number
   function SBatchNorm(net::Net, depth::Int; affine=true, dtype=Array{Float32}, momentum=.9, eps=1e-9)
      running_mean = dtype(zeros(1, 1, depth, 1))
      running_var = dtype(ones(1, 1, depth, 1))
      if affine
         push!(net, dtype(ones(1, 1, depth, 1)))
         push!(net, dtype(zeros(1, 1, depth, 1)))
         return new((length(net)-1):length(net), running_mean, running_var, momentum, eps)
      end
      new(length(net):length(net)-1, running_mean, running_var, momentum, eps)
   end
end

function forward(net, bn::SBatchNorm, x; mode=:train)
   assert(mode in [:train, :test])
   w = get_params(net, bn)
   affine = length(w) > 0
   tx = typeof(AutoGrad.getval(x))
   if typeof(bn.running_mean) !== tx
      bn.running_mean = tx(bn.running_mean)
   end
   if typeof(bn.running_var) !== tx
      bn.running_var = tx(bn.running_var)
   end
   x_hat = nothing
   if mode === :test
      x_hat = (x .- bn.running_mean) ./ sqrt(bn.running_var .+ bn.eps)
   else
      # Do the computation
      m = size(x, 1) * size(x, 2) * size(x, 4)
      mu = sum(x, (1, 2, 4)) ./ m
      x_mu = x .- mu
      sigma2 = sumabs2(x_mu, (1, 2, 4)) ./ m
      x_hat = x_mu ./ sqrt(sigma2 .+ bn.eps)
      # Update the running stats
      bn.running_mean = bn.momentum * bn.running_mean + (1 - bn.momentum) * AutoGrad.getval(mu)
      bn.running_var = bn.momentum * bn.running_var + (1 - bn.momentum) * AutoGrad.getval(sigma2)
   end
   affine ? (w[1] .* x_hat .+ w[2]) : x_hat
end

#Turn off weight decay
decay_range(layer::SBatchNorm) = 1:0
