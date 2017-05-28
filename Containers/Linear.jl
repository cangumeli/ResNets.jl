type Linear <: Layer
   range::Range{Int}
   function Linear(net::Net, input::Int, output::Int; bias=true, dtype=Array{Float32})
      s = length(net)+1
      stdv = 1 ./ sqrt(input)
      uniform(dims) = rand(dims) * 2stdv - stdv
      w = uniform((output, input))
      push!(net, dtype(w))
      if bias
         b = uniform((output, 1))
         push!(net, dtype(b))
      end
      e = length(net)
      new(s:e)
   end
end

function forward(net, lin::Linear, x)
   w = get_params(net, lin)
   #x = auto_convert(w, x, AutoGrad.getval)
   o1 = w[1] * ((ndims(x) > 2) ? mat(x) : x)
   if length(w) > 1
      return o1 .+ w[2]
   end
   return o1
end

function fill_weights!(net, lin::Linear, init::Function)
   w = get_params(net, lin)[1]
   copy!(w, typeof(w)(init(size(w))))
end

function fill_bias!(net, lin::Linear, init::Function)
   if length(lin.range) < 2
      warn("Linear layer doesn't have bias")
      return
   end
   w = get_params(net, lin)[2]
   copy!(w, typeof(w)(init(size(w))))
end

decay_range(l::Linear) = l.range[1:1] # only weights
