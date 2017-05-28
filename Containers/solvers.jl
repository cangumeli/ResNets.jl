abstract Solver

grad_taken(grad) = typeof(grad) !== Void

softloss(predict::Function) =
   (net, x, ygold) -> -sum(ygold .* logp(predict(net, x), 1)) ./ size(ygold, ndims(ygold))

softloss(net, y, ygold) =
    -sum(ygold .* logp(y, 1)) ./ size(ygold, ndims(ygold))

ssdloss(predict::Function) =
   (net, x, ygold) -> sumabs2(ygold .- predict(net, x)) ./ size(ygold, ndims(ygold))

ssdloss(net, y, ygold) =
   (net, x, ygold) -> sumabs2(ygold .- y) ./ size(ygold, ndims(ygold))


type SGD <: Solver
   lr::AbstractFloat
   momentum::AbstractFloat
   velocities::Any
   weight_decay::AbstractFloat
   nesterov::Bool
   #weight_decay_type::Symbol
   function SGD(lr; momentum=0.0, weight_decay=0.0, nesterov=false)#, weight_decay_type=:L2)
      #assert(weight_decay_type in (:L1, :L2))
      new(lr, momentum, nothing, weight_decay, nesterov)#, weight_decay_type)
    end
end

function update_net!(net, grads, solver::SGD, layers=nothing)
   if solver.velocities === nothing && solver.momentum > 0
      solver.velocities = []
      for i = 1:length(net)
         push!(solver.velocities, typeof(net[i])(zeros(size(net[i]))))
      end
   end
   # Assume l2 regularizatioN
   decay_indices = begin
      if solver.weight_decay == 0 || layers === nothing
         nothing
      else
         indices = Set{Int}()
         for l in layers
            drng = decay_range(l)
            if length(drng) > 0
               push!(indices, drng...)
            end
         end
         indices
      end
   end
   get_grad(i) =
      let cnd = (solver.weight_decay > 0 && (decay_indices === nothing || i in decay_indices))
         if cnd
            (solver.weight_decay * net[i] + grads[i])
         else
            grads[i]
         end
      end
   for i = 1:length(net)
      if ~grad_taken(grads[i])
         continue
      end
      if solver.momentum == 0
         net[i] -= solver.lr * get_grad(i)
      elseif solver.nesterov
         vprev = typeof(solver.velocities[i])(size(solver.velocities[i]))
         copy!(vprev, solver.velocities[i])
         solver.velocities[i] = solver.momentum * solver.velocities[i] - solver.lr * get_grad(i)
         net[i] += -solver.momentum * vprev + (1 + solver.momentum) * solver.velocities[i]
      else
         solver.velocities[i] = solver.momentum * solver.velocities[i] - solver.lr * get_grad(i)
         net[i] += solver.velocities[i]
      end
   end
end

# FIXME: remove this
#=function update_net!(net, grads, layers, solver::SGD)
   decay = solver.weight_decay
   solver.weight_decay = 0.0
   update_net!(net, grads, solver)
   solver.weight_decay = decay
   if decay > 0
      for l in layers
         rng = decay_range(l)
         for i = rng
            if grad_taken(grads[i])
               net[i] -= solver.lr decay * net[i]
            end
         end
      end
   end
end=#
