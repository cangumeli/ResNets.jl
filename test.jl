using Knet
include("Containers/NN.jl")

net = Net()

function init()
   d = Any[]
   push!(d, Linear(net, 100, 50))
   push!(d, Linear(net, 50, 10))
   return d
end

mlp = init()

function forward(net, x)
   o1 = relu(forward(net, mlp[1], x))
   o2 = forward(net, mlp[2], o1)
end


loss(net, x, y) = sumabs2(y-forward(net, x)) ./ size(y, 2)

lossgrad = grad(loss)

x = randn(100, 100)
y = randn(10, 100) * sin(x)
#opt = init_optim(net, x->Momentum(lr=.1, gamma=.9))
lr = 0.001

for i = 1:50000
   if i % 100 ==0; println("Iter ", i, ", Loss ", loss(net, x, y)); end
   g = lossgrad(net, x, y)
   for i = 1:length(g)
      net[i] -= lr * g[i]
   end
end
println("Loss ", loss(net, x, y))

lr = 0.1
net2 = Net()
l = Linear(net2, 100, 10)
forward2(net, x) = forward(net, l, x)
loss2(net, x, y) = sumabs2(y-forward2(net, x)) ./ size(y, 2)
lossgrad2 = grad(loss2)
println("Linear")
for i = 1:50000
   if i % 100 ==0; println("Iter ", i, ", Loss ", loss2(net2, x, y)); end
   g = lossgrad2(net2, x, y)
   for i = 1:length(g)
      net2[i] -= lr * g[i]
   end
end
println("Loss ", loss2(net2, x, y))
