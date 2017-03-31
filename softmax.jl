for p in ("Knet","ArgParse","Compat","GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

include("data.jl")
include("imgproc.jl")
using Knet
using ArgParse # To work with command line argumands
using Compat,GZip # Helpers to read the MNIST (Like lab-2)
using data


function main(args="")

    s = ArgParseSettings()
    @add_arg_table s begin
        ("--epochs"; arg_type=Int; default=10; help="number of epoch ")
        ("--batchsize"; arg_type=Int; default=100; help="size of minibatches")
        ("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--lr"; arg_type=Float64; default=0.5; help="learning rate")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
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
    ((xtrn, ytrn), (xtst, ytst)) = data.cifar10()
    mean_trn = imgproc.mean_subtract!(xtrn; mode=:pixel)
    xtst .-= mean_trn
    xtrn = mat(xtrn)
    xtst = mat(xtst)
    # println("Mean trn", size(mean_trn))

    # println(size(xtrn))
    #dtrn = minibatch(xtrn, ytrn, o[:batchsize])
    #dtst = minibatch(xtst, ytst, o[:batchsize])
    w = init_params(size(xtrn, 1), size(ytrn,1))
    # helper function to see how your training goes on.
    # report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
    train(w, (xtrn, ytrn), (xtst, ytst))
    return w
end

function init_params(ninputs, noutputs, winit=0.0001)
    #takes number of inputs and number of outputs(number of classes)
    #returns randomly generated W and b(must be zeros vector) params of softmax

    #start of step 2
    # YOUR CODE HERE
    W = winit * randn(ninputs, noutputs)
    b = zeros(noutputs, 1)
    return [W, b]
    #end of step 2
end

function next_batch(dtrn; batchsize=128)
   (x, y) = dtrn
   num_examples = size(x,2)
   sample = rand(UInt32, batchsize) % num_examples + 1
   return x[:, sample], y[:, sample]
end

function loss(w,x,ygold)
    scores = predict(w, x)
    # println("logp", size(logp(scores, 1)))
    # println("loss", size(ygold))
    return -sum(ygold .* logp(scores, 1)) ./ size(x, 2)
end

lossgradient =  grad(loss)# your code here [just 1 line]

function train(w, dtrn,dtst; lr=0.01, num_iters=10000, print_period=50)
   println("Iter/Training accuracy: ",0, "/", accuracy(w, dtrn)[1])
   println("Iter/Test accuracy: ",0, "/", accuracy(w, dtst)[1])
   println("")
   for i = 1:num_iters
      x, y = next_batch(dtrn)
      dw = lossgradient(w, x, y)
      for j = 1:length(dw)
         w[j] -= lr * dw[j]
      end
      if i % print_period == 0
         println("Iter/Training accuracy: ",i, "/", accuracy(w, dtrn)[1])
         println("Iter/Test accuracy: ",i, "/", accuracy(w, dtst)[1])
         println("")
      end
   end
    return w
end

function accuracy(w,dtst,pred=predict)
    ncorrect = 0
    ninstance = 0
    nloss = 0
    (X, Y) = dtst
    for i = 1:200:size(X,2)
      x = X[:, i:i+199]
      ygold = Y[:, i:i+199]
      ypred = predict(w, x)
      nloss += loss(w, x, ygold)
      ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
      ninstance += size(ygold, 2)
    end
    return (ncorrect/ninstance, nloss/ninstance)
end

#=
predict function takes model parameters (w) and data (x) and
makes forward calculation. Fill below function accordingly. It
should return #ofclasses x batchsize size matrix as a result.
Use ReLU nonlinearty at each layer except the last one.
=#
function predict(w,x)
   w[1]' * x .+ w[2]
end

main()
