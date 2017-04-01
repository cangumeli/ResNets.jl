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
        ("--epochs"; arg_type=Int; default=10; help="number of epoch ")
        ("--batchsize"; arg_type=Int; default=100; help="size of minibatches")
        # ("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--lr"; arg_type=Float64; default=0.15; help="learning rate")
        # ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
        ("--model"; arg_type=Int; default=0; help="model to train")
        ("--trn_buf"; arg_type=Bool; default=false; help="print a buffer of training accuracies")
        ("--print_acc"; arg_type=Bool; default=false; help="Print accuracy")
        ("--optim"; arg_type=String; default="sgd"; help="sgd or adam")
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

    # load the mnist data
    #=xtrnraw, ytrnraw, xtstraw, ytstraw = loaddata()

    xtrn = convert(Array{Float32}, reshape(xtrnraw ./ 255, 28*28, div(length(xtrnraw), 784)))
    ytrnraw[ytrnraw.==0]=10;
    ytrn = convert(Array{Float32}, sparse(convert(Vector{Int},ytrnraw),1:length(ytrnraw),one(eltype(ytrnraw)),10,length(ytrnraw)))

    xtst = convert(Array{Float32}, reshape(xtstraw ./ 255, 28*28, div(length(xtstraw), 784)))
    ytstraw[ytstraw.==0]=10;
    ytst = convert(Array{Float32}, sparse(convert(Vector{Int},ytstraw),1:length(ytstraw),one(eltype(ytstraw)),10,length(ytstraw)))
    # seperate it into batches.
    dtrn = minibatch4(xtrn, ytrn, o[:batchsize])
    dtst = minibatch4(xtst, ytst, o[:batchsize])=#
    dtr, dts = data.cifar10()
    (xtrn, ytrn) = dtr
    #println(ytrn)
    (xtst, ytst) = dts
    # mn = imgproc.mean_subtract!(xtrn;mode=:instance)
    mnt = mean(xtrn, 4)
    xtrn .-= mnt
    xtst .-= mnt
    # imgproc.mean_subtract!(xtst;mode=:instance)
    println(size(xtrn))
    println(size(xtst))
    #xtrn = xtrn[:, :, :, shuffle(1:size(xtrn, 4))]
    dtrn = minibatch(xtrn, ytrn, o[:batchsize])
    dtst = minibatch(xtst, ytst, o[:batchsize])
    # Main part of our training process
    model = nothing
    if o[:model] == 0
        model = model0
    else
        model = model1
    end
    train(dtrn, dtst; epochs=o[:epochs], model=model, buf=o[:trn_buf], print_report=o[:print_acc], lr=o[:lr], optimizer=o[:optim])
    #= @time for epoch=1:o[:epochs]
        train(w, dtrn,o[:lr])
        report(epoch)
    end=#
end

function minibatch(X, Y, bs)
   data = Any[]
   for i = 1:bs:size(X,4)
      mbatch = (X[:, :, :, i:i+bs-1], Y[:, i:i+bs-1])
      push!(data, mbatch)
   end
   return data
end

#=function minibatch(X, Y, bs; atype=Array{Float32})
    #takes raw input (X) and gold labels (Y)
    #returns list of minibatches (x, y)
    data = Any[]
    # YOUR CODE HERE
    for i=1:bs:size(X,2)
        mbatch = map(data->convert(atype, data), (X[:, i:i+bs-1], Y[:, i:i+bs-1]))
        push!(data, mbatch)
    end
    #YOUR CODE ENDS HERE
    return data
end=#

function minibatch4(x, y, batchsize; atype=Array{Float32})
    data = minibatch(x,y,batchsize; atype=atype)
    for i=1:length(data)
        (x,y) = data[i]
        data[i] = (reshape(x, (28,28,1,batchsize)), y)
    end
    return data
end

# Model for question 5
function model0(dtype=Array{Float32})
    function weights()
        weights = Any[]
        # Add conv layer weights
        push!(weights, -0.1 + 0.2 * rand(Float32, 5, 5, 3, 3))
        push!(weights, zeros(Float32, 1, 1, 3, 1))

        push!(weights, -0.1 + 0.2 * rand(Float32, 10, 16*16*3))
        push!(weights, zeros(Float32, 10, 1))
        #
        #=push!(weights, 0.0001 * rand(Float32, 10, 32*32*3))
        push!(weights, zeros(Float32, 10, 1))=#
        return map(data->convert(dtype, data), weights)
    end

    function predict(w, x)
        x1 = pool(relu(conv4(w[1], x; padding=2) .+ w[2]))
        return w[3] * mat(x1) .+ w[4]
        #return w[1] * mat(x) .+ w[2]
    end

    function loss(w,x,ygold)
        scores = predict(w, x)
        softloss = -sum(ygold .* logp(scores, 1)) ./ size(x, 4)
        #return softloss +  sum(w[1] .^2)
    end

    return weights, predict, loss

end

# My 100% accurate model
function model1(dtype=Array{Float32})
    function weights()
        weights = Any[]
        # Add conv layer weights
        push!(weights, -0.1 + 0.2 * rand(Float32, 5, 5, 3, 6))
        push!(weights, zeros(Float32, 1, 1, 6, 1))

        push!(weights, -0.1 + 0.2 * rand(Float32, 5, 5, 6, 16))
        push!(weights, zeros(Float32, 1, 1, 16, 1))

        # Add FC layer weights
        # push!(weights, -0.1 + 0.2 * rand(Float32, 10, 12*12*3))

        push!(weights, -0.1 + 0.2 * rand(Float32, 120, 5 * 5 * 16))
        push!(weights, zeros(Float32, 120, 1))

        push!(weights, -0.1 + 0.2 * rand(Float32, 84, 120))
        push!(weights, zeros(Float32, 84, 1))

        # push!(weights, -0.1 + 0.2 * rand(Float32, 10, 10*10*5))
        push!(weights, -0.1 + 0.2 * rand(Float32, 10, 84))
        push!(weights, zeros(Float32, 10, 1))
        return weights
        # return map(data->convert(dtype, data), weights)
    end

    function predict(w, x)

        x1 = pool(relu(conv4(w[1], x) .+ w[2]))
        x2 = pool(relu(conv4(w[3], x1) .+ w[4]))

        # println(size(x2))
        x3 = relu(w[5] * mat(x2) .+ w[6])
        x4 = relu(w[7] * x3 .+ w[8])
        w[9] * x4 .+ w[10]
        # return w[5] * mat(x2) .+ w[6]
    end

    #_, _, loss = model0()
    #=function loss(w,x,ygold)
        scores = predict(w, x)
        return -sum(ygold .* logp(scores, 1)) ./ size(x, 2)
    end=#
    function loss(w,x,ygold)
        scores = predict(w, x)
        return -sum(ygold .* logp(scores, 1)) ./ size(x, 4)
    end
    return weights, predict, loss
end

function modelgrad(model)
    _, _, loss = model()
    return grad(loss)
end

function train(dtrn, dtst; epochs=10, model=model1, print_report=true, buf=false, optimizer="sgd", lr=0.15)
    # helper function to see how your training goes on.
    # initalize weights of your model
    trn_buf = buf && Any[] # Buffer format for plotting
    weights, predict, loss = model()
    w = weights()
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn; model=model),:tst,accuracy(w,dtst; model=model)))
    if print_report
        report(0)
    end
    if buf
        push!(trn_buf, accuracy(w, dtrn; model=model)[1])
    end
    lossgrad = modelgrad(model)
    if optimizer == "adam"
        prms = map(x->Adam(), w)
    else
        prms = map(x->Sgd(lr=lr), w)
    end
    for epoch = 1:epochs
      shuffle!(dtrn)
        for (x, y) in dtrn
            g = lossgrad(w, x, y)
            update!(w, g, prms)
        end
        if print_report
            report(epoch)
        end
       if buf
           push!(trn_buf, accuracy(w, dtrn; model=model)[1])
       end
    end
    if buf
        println((:accs, trn_buf))
    end
    return w
end

function accuracy(w,dtst; model=model1)

    _, predict, loss = model()
    ncorrect = 0
    ninstance = 0
    nloss = 0
    # your code here
    for (x, ygold) in dtst
      # println("ygold", size(ygold))
        ypred = predict(w, x)
        nloss += loss(w, x, ygold)
        ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
        ninstance += size(ygold, 2)
    end
    # your code ends here
    # println("ninstance", ninstance)
    # println("")
    return (ncorrect / ninstance, nloss / length(dtst))
end
function loaddata()
    info("Loading MNIST...")
    xtrn = gzload("train-images-idx3-ubyte.gz")[17:end]
    xtst = gzload("t10k-images-idx3-ubyte.gz")[17:end]
    ytrn = gzload("train-labels-idx1-ubyte.gz")[9:end]
    ytst = gzload("t10k-labels-idx1-ubyte.gz")[9:end]
    return (xtrn, ytrn, xtst, ytst)
end

function gzload(file; path="$file", url="http://yann.lecun.com/exdb/mnist/$file")
    isfile(path) || download(url, path)
    f = gzopen(path)
    a = @compat read(f)
    close(f)
    return(a)
end
main()
