# Just to make sure you have installed all the packages that you will need in Lab-3
for p in ("Knet","ArgParse","Compat","GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet
using ArgParse # To work with command line argumands
using Compat,GZip # Helpers to read the MNIST (Like lab-2)
include("imgproc.jl")
include("data.jl")

function main(args="")
        #=
    In the macro, options and positional arguments are specified within a begin...end block
    by one or more names in a line, optionally followed by a list of settings.
    So, in  the below, there are five options: epoch,batchsize,hidden size of mlp,
    learning rate, weight initialization constant
    =#
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--iters"; arg_type=Int; default=150000; help="number of iterations ")
        ("--batchsize"; arg_type=Int; default=100; help="size of minibatches")
        # ("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--lr"; arg_type=Float32; default=Float32(0.001); help="learning rate")
        # ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
        ("--model"; arg_type=Int; default=0; help="model to train")
        ("--trn_buf"; arg_type=Bool; default=false; help="print a buffer of training accuracies")
        ("--print_period"; arg_type=Int; default=1000; help="Print accuracy in n iters")
        ("--optim"; arg_type=String; default="sgd"; help="sgd or adam")
        ("--momentum"; arg_type=Float32; default=Float32(0.9); help="momentum")
        ("--augment"; arg_type=Bool; default=true; help="Whether or not to augment the training data")
    end

        #=
    the actual argument parsing is performed via the parse_args function the result
    will be a Dict{String,Any} object.In our case, it will contain the keys "epochs",
    "batchsize", "hidden" and "lr", "winit" so that e.g. o["lr"] or o[:lr]
     will yield the value associated with the positional argument.
     For more information: http://argparsejl.readthedocs.io/en/latest/argparse.html
    =#
    o = parse_args(s; as_symbols=true)

    # Some global configs do not change here
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] = 123
    srand(o[:seed])
    #=dtr, dts = data.cifar10()
    (xtrn, ytrn) = dtr
    #println(ytrn)
    (xtst, ytst) = dts
    mnt = imgproc.mean_subtract!(xtrn;mode=:pixel)
    #mnt = mean(xtrn, 4)
    #xtrn .-= mnt
    xtst .-= mnt=#
    #imgproc.mean_subtract!(xtst;mode=:instance)
    dtrn, dtrn_, dval, dtst = loaddata(;augment=o[:augment], dtype=Array{Float32})
    println("Data is loaded...")
    # xtrn = xtrn[:, :, :, shuffle(1:size(xtrn, 4))]
    # dtrn = minibatch(xtrn, ytrn, o[:batchsize])
    # dtst = minibatch(xtst, ytst, o[:batchsize])
    w, model_ = train(dtrn_, dval; model=model, bsize=o[:batchsize],
      #buf=o[:trn_buf], print_report=o[:print_acc],
      lr=o[:lr], momentum=o[:momentum], iters=o[:iters], augmented=o[:augment],
      actual_trn = dtrn, print_period=o[:print_period]
   )
   println((:tst,accuracy(w,dtst; model=model_)))
end

function loaddata(;nval=5000, augment=true, dtype=Array{Float32})
   dtr, dts = data.cifar10()
   (xtrn, ytrn) = dtr
   (xtst, ytst) = dts
   # train-val split
   ntrain = size(ytrn, ndims(ytrn))
   sample = shuffle(1:ntrain)
   val = sample[1:nval]
   trn = sample[(nval+1):ntrain]
   xval, yval = xtrn[:, :, :, val], ytrn[:, val]
   xtrn, ytrn = xtrn[:, :, :, trn], ytrn[:, trn]

   mnt = imgproc.mean_subtract!(xtrn;mode=:pixel)
   xtst .-= mnt
   xval .-= mnt
   xtrn_augmented = xtrn
   if augment
      xtrn_augmented = zeros(36, 36, size(xtrn, 3), size(xtrn, 4))
      xtrn_augmented[3:34, 3:34, :, :] = xtrn
      xtrn_augmented = convert(dtype, xtrn_augmented)
   end
   println(size(xtrn))
   println(size(xtst))
   println(size(xval))
   return (xtrn, ytrn),(xtrn_augmented, ytrn), (xval, yval), (xtst, ytst)
end

# My 100% accurate model (hopefully)
function model(dtype=Array{Float32})

    function weights()
        weights = Any[]
        # Add conv layer weights
        #push!(weights, -0.1 + 0.2 * rand(Float32, 5, 5, 3, 6))
        push!(weights, randn(Float32, 5, 5, 3, 6) * sqrt(2 / (5*5*6)))
        push!(weights, zeros(Float32, 1, 1, 6, 1))

        # push!(weights, -0.1 + 0.2 * rand(Float32, 5, 5, 6, 16))
        push!(weights, randn(Float32, 5, 5, 6, 16) * sqrt(2/(5*5*16)))
        push!(weights, zeros(Float32, 1, 1, 16, 1))

        # Add FC layer weights
        # push!(weights, -0.1 + 0.2 * rand(Float32, 10, 12*12*3))

        #push!(weights, -0.1 + 0.2 * rand(Float32, 120, 5 * 5 * 16))
        push!(weights, randn(Float32, 120, 5 * 5 * 16) * sqrt(2/(5*5*16*120)))
        push!(weights, zeros(Float32, 120, 1))

        push!(weights, randn(Float32, 84, 120) * sqrt(2/(84*120)))
        push!(weights, zeros(Float32, 84, 1))

        # push!(weights, -0.1 + 0.2 * rand(Float32, 10, 10*10*5))
        push!(weights, -0.1 + 0.2 * rand(Float32, 10, 84))
        push!(weights, zeros(Float32, 10, 1))
        # return weights
        return map(data->convert(dtype, data), weights)
    end

    function predict(w, x)

        x1 = pool(relu(conv4(w[1], x) .+ w[2]))
        x2 = pool(relu(conv4(w[3], x1) .+ w[4]))

        # println(size(x2))
        x3 = relu(w[5] * mat(x2) .+ w[6])
        x4 = relu(w[7] * x3 .+ w[8])
        w[end-1] * x4 .+ w[end]
        # return w[5] * mat(x2) .+ w[6]
    end

    function loss(w,x,ygold)
        scores = predict(w, x)
        return -sum(ygold .* logp(scores, 1)) ./ size(x, 4)
    end
    return weights, predict, loss
end

function modelgrad(model)
    return grad(model[3])
end

function next_batch(x, y, bs; augmented=true)
   batch_indices = rand(1:size(x, 4), bs)
   x, y =  x[:, :, :, batch_indices], y[:, batch_indices]
   if augmented
      x_ = convert(typeof(x), zeros(32, 32, size(x,3), size(x,4)))
      for i = 1:bs
         rstart = rand(1:5)
         cstart = rand(1:5)
         x_[:, :, :, i] = x[rstart:rstart+31, cstart:cstart+31, :, i]
      end
      x = x_
   end
   return x, y
end

function train(dtrn, dtst; iters=15000, model=model, bsize=32, print_period=1000, lr=0.001, momentum=0.9, augmented=true, actual_trn=nothing)
   weights, predict, loss = model()
   model = (weights, predict, loss)
   report(iter)=println((:iter,iter,:trn,accuracy(w,actual_trn; model=model),:val,accuracy(w,dtst; model=model)))
    w = weights()
    lossgrad = modelgrad(model)
    prms = map(x->Momentum(lr=lr, gamma=momentum), w)
    for i = 1:iters
      x, y = next_batch(dtrn[1], dtrn[2], bsize; augmented=augmented)
      g = lossgrad(w, x, y)
      update!(w, g, prms)
      if i % print_period == 0
         report(i)
      end
   end
   return w, model
end

function accuracy(w,dtst; model=model)
    _, predict, loss = model
    ncorrect = 0
    ninstance = 0
    nloss = 0
    nloss_count = 0
    X, Y = dtst
    for i = 1:100:size(Y,2)
      x = X[:, :, :, i:i+99]
      ygold = Y[:, i:i+99]
      ypred = predict(w, x)
      nloss += loss(w, x, ygold)
      ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
      ninstance += size(ygold, 2)
      nloss_count += 1
    end
   return (ncorrect / ninstance, nloss / nloss_count)
end

main()
