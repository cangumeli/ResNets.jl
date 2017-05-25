type SGD
   lr::AbstractFloat
   momentum::AbstractFloat
   velocities::Any
   weight_decay::AbstractFloat
   #weight_decay_type::Symbol
   function SGD(lr; momentum=0.0, weight_decay=0.0)#, weight_decay_type=:L2)
      #assert(weight_decay_type in (:L1, :L2))
      new(lr, momentum, nothing, weight_decay)#, weight_decay_type)
    end
end

#TODO: support undefs
#TODO: add gradient conservation
function update_net!(net, grads, solver::SGD)
   if solver.velocities === nothing && solver.momentum > 0
      solver.velocities = []
      for i = 1:length(net)
         push!(solver.velocities, typeof(net[i])(zeros(size(net[i]))))
      end
   end
   # Assume l2 regularizatioN
   get_grad(i) = (solver.weight_decay > 0) ? (solver.weight_decay * net[i] + grads[i]) : grads[i]
   for i = 1:length(net)
      if solver.momentum == 0
         net[i] -= solver.lr * get_grad(i)
      else
         solver.velocities[i] = solver.momentum * solver.velocities[i] + solver.lr * get_grad(i)
         net[i] -= solver.velocities[i]
      end
   end
end

function update_net!(net, grads, layers, solver::SGD)
   decay = solver.weight_decay
   solver.weight_decay = 0.0
   update_net!(net, grads, solver)
   solver.weight_decay = decay
   if decay > 0
      for l in layers
         rng = decay_range(l)
         for i = rng
            net[i] -= decay * net[i]
         end
      end
   end
end
