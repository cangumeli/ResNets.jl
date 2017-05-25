include("resnet_con.jl")

Config, Group = ResNetConfig, GroupConfig

function resnet_cifar(n; num_classes=10, pre=false)
   groups = [Group(16, 16, n, 1), Group(16, 32, n, 2), Group(32, 64, n, 2)]
   cfg = Config(
      pre ? pre_basic_block : basic_block,
      groups,
      pre ? pre_basic_block_forward : basic_block_forward
   )
   init_conv(net) = Conv4(net, 3, 3, 3, 16; padding=1, bias=false)
   return create_resnet(cfg, init_conv, num_classes)
end

function preresnet_cifar_bottleneck(n; num_classes=10)
   groups = [Group(16, 64, n, 1), Group(64, 128, n, 2), Group(128, 256, n, 2)]
   cfg = Config(pre_bottleneck_block, groups, pre_bottleneck_block_forward)
   init_conv(net) = Conv4(net, 3, 3, 3, 16; padding=1, bias=false)
   return create_resnet(cfg, init_conv, num_classes)
end

resnet110(;num_classes=10, pre=false) = resnet_cifar(18; num_classes=num_classes, pre=pre)
resnet164(;num_classes=10) = preresnet_cifar_bottleneck(18; num_classes=num_classes)
resnet1001(;num_classes=10) = preresnet_cifar_bottleneck(111; num_classes=num_classes)

function imagenet_start_ops()
   init_conv(net) = Conv4(net, 7, 7, 3, 64; stride=2, padding=3, bias=false)
   init_pool(x) = pool(x; window=(3, 3), stride=2)
   return init_conv, init_pool
end

function resnet101()
   groups = [
      Group(64  , 256 , 3,  1),
      Group(256 , 512 , 4 , 2),
      Group(512 , 1024, 23, 2),
      Group(1024, 2048, 3 , 2),
   ]
   cfg = Config(bottleneck_block, groups, bottleneck_block_forward)
   init_conv, init_pool = imagenet_start_ops()
   return create_resnet(cfg, init_conv, 1000, init_pool)
end

#=net, layers, predict = resnet101()
println("Length of net ", length(net))
# println("Sizes of the net ", map(x->size(x), net))
println("Layers ", map(x->typeof(x), layers))
output = predict(net, randn(Float32, 224, 224, 3, 8))
println("Output ", size(output))=#

net, layers, predict = resnet110(;pre=true)
output = predict(net, randn(Float32, 32, 32, 3, 64); debug=true)
println(size(output))

net, layers, predict = resnet1001()
println("Length of net ", length(net))
#println("Layers ", map(x->typeof(x), layers))
#println("Forward function ", predict)
output = predict(net, randn(Float32, 32, 32, 3, 64); debug=true)
println(size(output))
#println(output)
