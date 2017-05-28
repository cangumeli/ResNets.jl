using Knet

# Abstract datatype for containers
abstract Layer

# The network as a parameter array
typealias Net Array{Any, 1}

type Variable
   index::Int
   Variable(net, w) = push!(net, w)
end

get_value(net, variable::Variable) = net[layer.index]
# Implicitly force layers to declare range
get_params(net, layer::Layer) = net[layer.range]

net_dims(net, layer::Layer) = map(x->size(x), net[layer.range])

function gpu!(net::Net; dtype=Float32)
   for i = 1:length(net)
      net[i] = KnetArray{dtype}(net[i])
   end
end

function cpu!(net::Net; dtype=Float32)
   for i = 1:length(net)
      net[i] = Array{dtype}(net[i])
   end
end

# Returns the weights to be decayed by optimizers
decay_range(layer::Layer) = layer.range

#end
