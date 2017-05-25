include("Containers/NN.jl")
include("Containers/solvers.jl")
include("data.jl")
using Knet

function init_model(dtype=Array{Float32})
   net = Net()
   layers = Dict{Any, Any}()
   layers[:c1] = Conv4(net, 3, 3, 3, 16; padding=1, bias=false)
   layers[:bn1] = SBatchNorm(net, 16)
   layers[:c2] = Conv4(net, 3, 3, 16, 32; padding=1, bias=false)
   layers[:bn2] = SBatchNorm(net, 32)
   layers[:fc1] = Linear(net, 8 * 8 * 32, 100; bias=false)
   layers[:bn3] = BatchNorm(net, 100)
   layers[:output] = Linear(net, 100, 10)
   return net, layers
end

net, layers = init_model()
if Knet.gpu() >= 0
   gpu!(net)
end

function predict(net, x; mode=:train)
   convbn(cl, bnl, x) = forward(net, layers[bnl], forward(net, layers[cl], x); mode=mode)
   fcbn(l, bnl, x) = forward(net, layers[bnl], forward(net, layers[l], x); mode=mode)
   o1 = pool(relu(convbn(:c1, :bn1, x)))
   o2 = pool(relu(convbn(:c2, :bn2, o1)))
   o3 = relu(fcbn(:fc1, :bn3, o2))
   return forward(net, layers[:output], o3)
end

loss(net, x, ygold; mode=:train) = -sum(ygold .* logp(predict(net, x; mode=mode), 1)) ./ size(ygold, 2)
result_loss(net, y, ygold) = -sum(ygold .* logp(y, 1)) ./ size(ygold, 2)
lossgrad = grad(loss)

function accuracy(net, dtst; dtype=Array{Float32})
    println("Computing Accuracy...")
    ncorrect = 0
    ninstance = 0
    nloss = 0
    nloss_count = 0
    X, Y = dtst
    for i = 1:100:size(Y,2)
        x = convert(dtype, X[:, :, :, i:i+99])
        ygold = convert(dtype, Y[:, i:i+99])
        if i % 1000 == 0
            println("Accuracy iter ", i)
        end
        #println("x from the accuracy ", size(x))
        ypred = predict(net, x; mode=:test)
        nloss += result_loss(net, ypred, ygold) # diminish the side effects
        ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
        ninstance += size(ygold, 2)
        nloss_count += 1
    end
    println(ncorrect, " ", ninstance," ", nloss, " ", nloss_count)
    return (ncorrect / ninstance, nloss / nloss_count)
end


function loaddata()
   dtr, dts = data.cifar10()
   (xtrn, ytrn) = dtr
   (xtst, ytst) = dts
   mnt = mean(xtrn, (3, 4))
   xtrn .-= mnt
   xtst .-= mnt
   return (xtrn, ytrn), (xtst, ytst)
end

function next_batch(x, y; dtype=Array{Float32}, bs=128)
   batch_indices = rand(1:size(x, 4), bs)
   x_, y_ =  x[:, :, :, batch_indices], y[:, batch_indices]
   return dtype(x_), dtype(y_)
end

function train(;iters=10000, bsize=128, print_period=50, lr=0.1)
   dtrn, dtst = loaddata()
   solver = SGD(lr; momentum=.9, weight_decay=1e-4)
   for i = 1:iters
      println("Iter ", i)
      if i == 1 || i % print_period == 0
         # println("Train Accuracy ", accuracy(net, dtrn))
         println("Test Accuracy ", accuracy(net, dtst))
      end
      if i % 1000 == 0
         solver.lr = max(lr * 0.1, 0.001)
      end
      x, y = next_batch(dtrn[1], dtrn[2]; bs=bsize)
      g = lossgrad(net, x, y)
      update_net!(net, g, [layers[k] for k in keys(layers)], solver)
   end
end

train()
