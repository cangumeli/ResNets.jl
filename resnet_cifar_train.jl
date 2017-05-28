include("data.jl")
include("imgproc.jl")
include("resnet_con_oop.jl")
include("Containers/solvers.jl")

function resnet_cifar(depth::Int, num_classes=10; pre=false)
   n = Int((depth - 2) / 6)
   groups = [GroupConfig(16, 16, n, 1), GroupConfig(16, 32, n, 2), GroupConfig(32, 64, n, 2)]
   config = ResNetConfig(pre ? BasicBlockPre : BasicBlock, groups)
   return create_resnet(
      net->Conv4(net, 3, 3, 3, 16; padding=1, bias=false),
      config,
      num_classes
   )
end

function preresnet_cifar_deep(depth::Int, num_classes=10)
    n = Int((depth - 2) / 9)
    groups = [GroupConfig(16, 64, n, 1), GroupConfig(64, 128, n, 2), GroupConfig(128, 256, n, 2)]
    config = ResNetConfig(BottleneckBlockPre, groups)
    return create_resnet(
        net->Conv4(net, 3, 3, 3, 16; padding=1, bias=false),
        config,
        num_classes
    )
end

function loaddata(;nval=5000, augment=true, dtype=Array{Float32})
    println("Loading data...")
    dtr, dts = data.cifar10()
    println("Data is read...")
    (xtrn, ytrn) = dtr
    (xtst, ytst) = dts
    # train-val split
    xval, yval = nothing, nothing
    if nval > 0
        ntrain = size(ytrn, ndims(ytrn))
        sample = shuffle(1:ntrain)
        val = sample[1:nval]
        trn = sample[(nval+1):ntrain]
        xval, yval = xtrn[:, :, :, val], ytrn[:, val]
        xtrn, ytrn = xtrn[:, :, :, trn], ytrn[:, trn]
    end
    mnt = mean(xtrn, (3, 4))
    xtrn .-= mnt
    xtst .-= mnt
    if nval > 0
        xval .-= mnt
    end
    xtrn_augmented = xtrn
    if augment
        xtrn_augmented = zeros(40, 40, size(xtrn, 3), size(xtrn, 4))
        xtrn_augmented[5:36, 5:36, :, :] = xtrn
        xtrn_augmented = convert(dtype, xtrn_augmented)
    end
    println(size(xtrn))
    println(size(xtst))
    println(nval > 0 ? size(xval) : 0)
    return (xtrn, ytrn),(xtrn_augmented, ytrn), (xval, yval), (xtst, ytst)
end

function next_batch(x, y, bs; dtype=Array{Float32}, augmented=true)
    batch_indices = rand(1:size(x, 4), bs)
    x, y =  x[:, :, :, batch_indices], y[:, batch_indices]
    if augmented
        x_ = convert(typeof(x), zeros(32, 32, size(x,3), size(x,4)))
        for i = 1:bs
            rstart = rand(1:9)
            cstart = rand(1:9)
            p = rand()
            if p <= .5
                x_[:, :, :, i] = imgproc.flip_horizontal(x[rstart:rstart+31, cstart:cstart+31, :, i])
            else
                x_[:, :, :, i] = x[rstart:rstart+31, cstart:cstart+31, :, i]
            end
        end
        x = x_
    end
    return convert(dtype, x), convert(dtype, y)
end

function accuracy(w, dtst, predict; dtype=Array{Float32}, bmode=:test)
    println("Computing Accuracy...")
    ncorrect = 0
    ninstance = 0
    nloss = 0
    nloss_count = 0
    X, Y = dtst
    for i = 1:100:size(Y,2)
        x = convert(dtype, X[:, :, :, i:i+99])
        ygold = convert(dtype, Y[:, i:i+99])
        #=println(typeof(x))
        println(typeof(y))=#
        if i % 1000 == 0
            println("Accuracy iter ", i)
        end
        ypred = predict(w, x; mode=bmode)
        nloss += softloss(w, ypred, ygold) # diminish the side effects
        ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
        ninstance += size(ygold, 2)
        nloss_count += 1
    end
    println(ncorrect, " ", ninstance," ", nloss, " ", nloss_count)
    return (ncorrect / ninstance, nloss / nloss_count)
end

# TODO: Support CIFAR100, model serialization
for p in ("ArgParse",)
   if Pkg.installed(p) == nothing; Pkg.add(p); end
end
using ArgParse
s = ArgParseSettings()
ftype = Float32
@add_arg_table s begin
    "--model"; arg_type=String; default="standard"; help="Model type, options: standard, standard_pre, deep_pre"
    "--depth"; arg_type=Int; default=110; help="Model depth"
    "--iters"; arg_type=Int; default=64000; help="Number of training iterations"
    "--pp"; arg_type=Int; default=400; help="Print accuracy at each pp iterations"
    "--lr"; arg_type=ftype; default=ftype(0.1); help="The initial learning rate"
    "--momentum"; arg_type=ftype; default=ftype(0.9); help="Momentum of Sgd"
    "--wdecay"; arg_type=ftype; default=ftype(1e-4); help="L2 decay of Sgd"
    "--lrdecay"; arg_type=Array{Int, 1}; default=[32000, 48000]; help="The iterations of learning rate decay"
    "--bsize"; arg_type=Int; default=128; help="Minibatch size"
    "--warmup"; arg_type=Int; default=0; help="Number of iterations with lr / 10"
end

o = parse_args(s; as_symbols=true)
println("opts=",[(k,v) for (k,v) in o]...)

assert(o[:model] in ["standard", "standard_pre", "deep_pre"])
assert(o[:model] === "standard" && (o[:depth] - 2) % 6 === 0 || (o[:depth] - 2) % 9 === 0)

net, layers, predict = let pd = o[:model] === "deep_pre"
    if pd
        preresnet_cifar_deep(o[:depth])
    else
        resnet_cifar(o[:depth]; pre=o[:model] === "standard_pre")
    end
end

dtype_trn, dtype_init = Array{Float32}, Array{Float32}
if Knet.gpu() >= 0
    println("Transfering model to gpu...")
    gpu!(net)
    println("Transfer completed...\n")
    dtype_trn = KnetArray{Float32}
end

lossgrad = grad(softloss(predict))
solver = SGD((o[:warmup] > 0) ? 0.1o[:lr] : o[:lr]; momentum=o[:momentum], weight_decay=o[:wdecay])
dtrn, dtrn_, dval, dtst = loaddata(;nval=0, dtype=dtype_init)
for i = 1:o[:iters]
   if i === 1 || i % 50 === 0; println("Iter ", i); end
   x, y = next_batch(dtrn_[1], dtrn_[2], o[:bsize]; dtype=dtype_trn)
   g = lossgrad(net, x, y)
   update_net!(net, g, solver)
   if i === o[:warmup]
      solver.lr = o[:lr]
   end
   if i in o[:lrdecay]
      solver.lr *= 0.1
   end
   if i % o[:pp] === 0 || i === 1
       println("Training accuracy ", accuracy(net, dtrn, predict; dtype=dtype_trn))
       println("Test accuracy ", accuracy(net, dtst, predict; dtype=dtype_trn))
   end
end
