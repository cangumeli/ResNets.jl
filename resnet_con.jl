include("Containers/NN.jl")

type GroupConfig
   input::Int
   output::Int
   repeat::Int
   stride::Int
end

type ResNetConfig
   block::Any
   groups::Array{GroupConfig, 1}
   block_forward::Any
end

shortcut(net, input, output; stride=1) = Conv4(net, 1, 1, input, output; stride=stride, bias=false)

conv3x3(net, input, output; stride=1) = Conv4(net, 3, 3, input, output; padding=1, stride=stride, bias=false)

function add_shortcut!(layers, net, input, output; stride=1)
   if input !== output
      layers[:shortcut] = shortcut(net, input, output; stride=stride)
      layers[:shortcut_bn] = SBatchNorm(net, output)
   end
end

function basic_block(net, input, output; stride=1)
   layers = Dict{Symbol, Layer}()
   add_shortcut!(layers, net, input, output; stride=stride)
   layers[:conv1] = conv3x3(net, input, output; stride=stride)
   layers[:bn1] = SBatchNorm(net, output)
   layers[:conv2] = conv3x3(net, output, output)
   layers[:bn2] = SBatchNorm(net, output)
   return layers
end

function bottleneck_block(net, input, output; stride=1)
   layers = Dict{Symbol, Layer}()
   add_shortcut!(layers, net, input, output; stride=stride)
   n = Int(output / 4)
   layers[:conv1] = Conv4(net, 1, 1, input, n)
   layers[:bn1] = SBatchNorm(net, n)
   layers[:conv2] = conv3x3(net, n, n; stride=stride)
   layers[:bn2] = SBatchNorm(net, n)
   layers[:conv3] = Conv4(net, 1, 1, n, output)
   layers[:bn3] = SBatchNorm(net, output)
   return layers
end

function shortcut_forward(net, layers, x; mode=:train)
   o = x
   if haskey(layers, :shortcut)
      o = forward(net, layers[:shortcut], o)
   end
   if haskey(layers, :shortcut_bn)
      o = forward(net, layers[:shortcut_bn], o; mode=mode)
   end
   return o
end

function basic_block_forward(net, layers, x; mode=:train)
   o1 = forward(net, layers[:conv1], x)
   o2 = relu(forward(net, layers[:bn1], o1; mode=mode))
   o3 = forward(net, layers[:conv2], o2)
   o4 = forward(net, layers[:bn2], o3; mode=mode)
   o0 = shortcut_forward(net, layers, x; mode=mode)
   return relu(o4 + o0)
end

function bottleneck_block_forward(net, layers, x; mode=:train)
   convbn(conv, bn, x) = forward(net, layers[bn], forward(net, layers[conv], x); mode=mode)
   o1 = relu(convbn(:conv1, :bn1, x))
   o2 = relu(convbn(:conv2, :bn2, o1))
   o3 = convbn(:conv3, :bn3, o2)
   o0 = shortcut_forward(net, layers, x; mode=mode)
   return relu(o3 + o0)
end

function pre_basic_block(net, input, output; stride=1)
   layers = Dict{Symbol, Layer}()
   add_shortcut!(layers, net, input, output; stride=stride)
   layers[:bn1] = SBatchNorm(net, input)
   layers[:conv1] = conv3x3(net, input, output; stride=stride)
   layers[:bn2] = SBatchNorm(net, output)
   layers[:conv2] = conv3x3(net, output, output)
   return layers
end

function pre_basic_block_forward(net, layers, x; mode=:train, first=false)
   bnrc(bn, cv, x) = relu(forward(net, layers[cv], forward(net, layers[bn], x; mode=mode)))
   o1 = let
      if first
         x = relu(forward(net, layers[:bn1], x; mode=mode))
         forward(net, layers[:conv1], x)
      else
         bnrc(:bn1, :conv1, x)
      end
   end
   o2 = bnrc(:bn2, :conv2, o1)
   o0 = shortcut_forward(net, layers, x; mode=mode)
   return o2 + o0
end

function pre_bottleneck_block(net, input, output; stride=1)
   layers = Dict{Symbol, Layer}()
   add_shortcut!(layers, net, input, output; stride=stride)
   n = Int(output / 4)
   layers[:bn1] = SBatchNorm(net, input)
   layers[:conv1] = Conv4(net, 1, 1, input, n)
   layers[:bn2] = SBatchNorm(net, n)
   layers[:conv2] = conv3x3(net, n, n; stride=stride)
   layers[:bn3] = SBatchNorm(net, n)
   layers[:conv3] = Conv4(net, 1, 1, n, output)
   return layers
end

function pre_bottleneck_block_forward(net, layers, x; mode=:train, first=false)
   bnrc(bn, cv, x) = relu(forward(net, layers[cv], forward(net, layers[bn], x; mode=mode)))
   o1 = let
      if first
         x = relu(forward(net, layers[:bn1], x; mode=mode))
         forward(net, layers[:conv1], x)
      else
         bnrc(:bn1, :conv1, x)
      end
   end
   o2 = bnrc(:bn2, :conv2, o1)
   o3 = bnrc(:bn3, :conv3, o2)
   o0 = shortcut_forward(net, layers, x; mode=mode)
   return o3 + o0
end

# TODO: consider to divide pre and standard
function create_resnet(config::ResNetConfig, first_conv::Function, num_classes::Int, first_pool=nothing, pre=false)
   net = Net()
   layers = []
   if first_pool !== nothing || ~pre
      push!(layers, Dict([:conv=>first_conv(net), :bn=>SBatchNorm(net, config.groups[1].input)]))
   else # cifar preact models
      push!(layers, Dict([:conv=>first_conv(net)]))
   end
   for cf in config.groups
      push!(layers, config.block(net, cf.input, cf.output; stride=cf.stride))
      for i = 2:cf.repeat
         push!(layers, config.block(net, cf.output, cf.output))
      end
   end
   push!(layers, Linear(net, config.groups[end].output, num_classes))

   function predict(net, x; mode=:train, debug=false)
      o = let
         if pre && (first_pool === nothing)
            forward(net, layers[1][:conv], x)
         else
            relu(forward(net, layers[1][:bn], forward(net, layers[1][:conv], x); mode=mode))
         end
      end
      if first_pool !== nothing
         o = first_pool(o)
      end
      for i = 2:(length(layers) - 1)
         o = let add_first = pre && (first_pool !== nothing)
            if add_first
               config.block_forward(net, layers[i], o; mode=mode, first=i===2)
            else
               config.block_forward(net, layers[i], o; mode=mode)
            end
         end
      end
      if debug # ImageNet => 7x7, Cifar => 8x8
         println("Size of the feature map: ", size(o)[1:2])
      end
      return forward(net, layers[end], pool(o; window=size(o)[1:2], mode=2))
   end

   return net, layers, predict
end
