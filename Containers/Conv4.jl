type Conv4 <: Layer
   range::Range{Int}
   stride::Int
   padding::Int
   function Conv4(net::Net, height::Int, width::Int, input::Int, output::Int;
      bias=true, dtype=Array{Float32}, stride=1, padding=0)
      s = length(net)+1
      w = dtype(randn(height, width, input, output) *
            sqrt(2.0 / (height * width * output)))
      push!(net, w)
      if bias
          b = dtype(zeros(1, 1, output, 1))
          push!(net, b)
      end
      new(s:length(net), stride, padding)
   end
end

function forward(net, cn::Conv4, x)
   w = get_params(net, cn)
   o1 = conv4(w[1], x; padding=cn.padding, stride=cn.stride)
   if length(w) > 1
      return o1 .+ w[2]
   end
   return o1
end

# init: ((height, width, input, output))
function fill_weights!(net, cn::Conv4, init::Function)
   weight = get_params(net, cn)[1]
   copy!(weight, typeof(weight)(init(size(weight))))
end

function fill_bias!(net, cn::Conv4, init::Function)
   params = get_params(net, cn)
   if length(params) < 2
      error("Layer doesn't have bias")
   end
   bias = params[2]
   copy!(bias, typeof(bias)(init(size(bias))))
end

decay_range(l::Conv4) = l.range[1:1]
