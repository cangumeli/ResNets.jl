include("Containers/NN.jl")

abstract ResNetBlock

type GroupConfig
   input::Int
   output::Int
   repeat::Int
   stride::Int
end

type ResNetConfig
   Block
   groups::Array{GroupConfig, 1}
end

type Shortcut
   conv
   bn
   function Shortcut(net, input, output; stride=1)
      if input === output
         return new(nothing, nothing)
      end
      new(
         Conv4(net, 1, 1, input, output; stride=stride, bias=false),
         SBatchNorm(net, output)
      )
   end
end

function forward(net, shortcut::Shortcut, x; mode=:train)
   o = x
   if shortcut.conv !== nothing
      o = forward(net, shortcut.conv, o)
   end
   if shortcut.bn !== nothing
      o = forward(net, shortcut.bn, o; mode=mode)
   end
   return o
end


conv3x3(net, input, output; stride=1) = Conv4(net, 3, 3, input, output; padding=1, stride=stride, bias=false)

type BasicBlock <: ResNetBlock
   conv1::Conv4
   bn1::SBatchNorm
   conv2::Conv4
   bn2::SBatchNorm
   shortcut::Shortcut
   BasicBlock(net::Net, input::Int, output::Int; stride=1) = new(
      conv3x3(net, input, output; stride=stride),
      SBatchNorm(net ,output),
      conv3x3(net, output, output),
      SBatchNorm(net, output),
      Shortcut(net, input, output; stride=stride),
   )
end

function forward(net, block::BasicBlock, x; mode=:train)
   convbn(conv, bn, x) = forward(net, bn, forward(net, conv, x); mode=mode)
   o1 = relu(convbn(block.conv1, block.bn1, x))
   o2 = convbn(block.conv2, block.bn2, o1)
   o0 = forward(net, block.shortcut, x)
   relu(o0 + o2)
end

conv1x1(net, input, output) = Conv4(net, 1, 1, input, output; bias=false)

type BottleneckBlock <: ResNetBlock
   conv1::Conv4
   bn1::SBatchNorm
   conv2::Conv4
   bn2::SBatchNorm
   conv3::Conv4
   bn3::SBatchNorm
   shortcut::Shortcut
   BottleneckBlock(net::Net, input::Int, output::Int; stride=1) =
      let n = Int(output / 4)
         new(
            conv1x1(net, input, n), SBatchNorm(net, n),
            conv3x3(net, n, n; stride=stride), SBatchNorm(net, n),
            conv1x1(net, n, output), SBatchNorm(net, output),
            Shortcut(net, input, output; stride=stride)
         )
      end
end

function forward(net, block::BottleneckBlock, x; mode=:train)
   convbn(conv, bn, x) = forward(net, bn, forward(net, conv, x); mode=mode)
   o1 = relu(convbn(block.conv1, block.bn1, x))
   o2 = relu(convbn(block.conv2, block.bn2, o1))
   o3 = convbn(block.conv3, block.bn3, o2)
   o0 = forward(net, block.shortcut, x)
   relu(o3 + o0)
end

type BasicBlockPre <: ResNetBlock
   bn1
   conv1::Conv4
   bn2::SBatchNorm
   conv2::Conv4
   shortcut::Shortcut
   BasicBlockPre(net::Net, input::Int, output::Int; stride=1, first=false) =new(
      first ? nothing : SBatchNorm(net, input),
      conv3x3(net, input, output; stride=stride),
      SBatchNorm(net, output),
      conv3x3(net, output, output),
      Shortcut(net, input, output; stride=stride)
   )
end

function forward(net, block::BasicBlockPre, x; mode=:train)
   o1 = x
   if block.bn1 !== nothing
      o1 = relu(forward(net, block.bn1, x; mode=mode))
   end
   o2 = forward(net, block.conv1, o1)
   o3 = relu(forward(net, block.bn2, o2; mode=mode))
   o4 = forward(net, block.conv2, o3)
   o0 = forward(net, block.shortcut, x; mode=mode)
   return o4 + o0
end

type BottleneckBlockPre <: ResNetBlock
   bn1
   conv1::Conv4
   bn2::SBatchNorm
   conv2::Conv4
   bn3::SBatchNorm
   conv3::Conv4
   shortcut::Shortcut
   BottleneckBlockPre(net::Net, input::Int, output::Int; stride=1, first=false) =
      let n = Int(output / 4)
         new(
            first ?  nothing : SBatchNorm(net, input),
            conv1x1(net, input, n),
            SBatchNorm(net, n), conv3x3(net, n, n; stride=stride),
            SBatchNorm(net, n), conv1x1(net, n, output),
            Shortcut(net, input, output; stride=stride)
         )
      end
end

function forward(net, block::BottleneckBlockPre, x; mode=:train)
   bnconv(bn, conv, x) =
      let conv_in = (bn === nothing) ? x : relu(forward(net, bn, x; mode=mode))
         forward(net, conv, conv_in)
      end
   o1 = bnconv(block.bn1, block.conv1, x)
   o2 = bnconv(block.bn2, block.conv2, o1)
   o3 = bnconv(block.bn3, block.conv3, o2)
   o0 = forward(net, block.shortcut, x; mode=mode)
   return o3 + o0
end

type ResNetStart
   conv::Conv4
   pool # x -> pool(x)
   bn
   ResNetStart(net, create_conv::Function, pool=nothing ;bnorm=true) =
      let conv = create_conv(net)
         new(
            conv,
            pool,
            bnorm ? SBatchNorm(net, net_dims(net, conv)[1][end]) : nothing
         )
      end
end

function forward(net, start::ResNetStart, x; mode=:train)
   o = forward(net, start.conv, x)
   if start.bn !== nothing
      o = relu(forward(net, start.bn, o; mode=mode))
   end
   if start.pool !== nothing
      o = start.pool(o)
   end
   return o
end

type ResNetFinal
   bn
   fc::Linear
   ResNetFinal(net::Net, num_features::Int, num_classes::Int; bnorm=false) = new(
      bnorm ? SBatchNorm(net, num_features) : nothing,
      Linear(net, num_features, num_classes)
   )
end

function forward(net, block::ResNetFinal, x; mode=:train)
   o = x
   if block.bn !== nothing
      o = relu(forward(net, block.bn, o; mode=mode))
   end
   o = pool(o; mode=2, window=size(o)[1:2])
   return forward(net, block.fc, o)
end

function create_resnet(start_conv::Function, config::ResNetConfig, num_classes::Int, start_pool=nothing)
   net = Net()
   layers = []
   pre = config.Block in [BasicBlockPre, BottleneckBlockPre]
   println("Pre: ", pre) 
   push!(layers, ResNetStart(net, start_conv, start_pool; bnorm=~pre))
   for cf in config.groups
      push!(layers, config.Block(net, cf.input, cf.output; stride=cf.stride))
      for i = 2:cf.repeat
         push!(layers, config.Block(net, cf.output, cf.output))
      end
   end
   push!(layers, ResNetFinal(net, config.groups[end].output, num_classes))

   function predict(net, x; mode=:train, debug=false)
      o = x
      for l in layers
         if debug; println(size(o)); end
         o = forward(net, l, o; mode=mode)
      end
      return o
   end

   return net, layers, predict
end
