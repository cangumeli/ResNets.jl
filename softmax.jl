for p in ("Knet","ArgParse","Compat","GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

include("data.jl")
include("imgproc.jl")
using Knet


function main()
    dtrn, dval, dtst = loaddata()
    w = init_params(size(dtrn[1], 1), size(dtrn[2],1))
    train(w, dtrn, dval; num_iters=1000)
    println("Final accuracy ", accuracy(w, dtst)[1])
    return w
end

function loaddata(nval=5000)
   ((xtrn, ytrn), (xtst, ytst)) = data.cifar10()
   ntrain = size(xtrn, 4)
   order = shuffle(1:ntrain)
   (xval, yval) = (xtrn[:, :, :, order[1:nval]], ytrn[:, order[1:nval]])
   (xtrn, ytrn) = (xtrn[:, :, :, order[(nval+1):ntrain]], ytrn[:, order[(nval+1):ntrain]])
   mean_trn = imgproc.mean_subtract!(xtrn; mode=:pixel)
   xtst .-= mean_trn
   xval .-= mean_trn
   xtrn = mat(xtrn)
   xval = mat(xval)
   xtst = mat(xtst)
   return (xtrn, ytrn), (xval, yval), (xtst, ytst)
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

function train(w, dtrn, dval; lr=0.01, num_iters=10000, print_period=50)
   println("Iter/Training accuracy: ",0, "/", accuracy(w, dtrn)[1])
   println("Iter/Test accuracy: ",0, "/", accuracy(w, dval)[1])
   println("")
   for i = 1:num_iters
      x, y = next_batch(dtrn)
      dw = lossgradient(w, x, y)
      for j = 1:length(dw)
         w[j] -= lr * dw[j]
      end
      if i % print_period == 0
         println("Iter/Training accuracy: ",i, "/", accuracy(w, dtrn)[1])
         println("Iter/Val accuracy: ",i, "/", accuracy(w, dval)[1])
         println("")
      end
   end
    return w
end

function accuracy(w,dtst,pred=predict,bsize=200)
    ncorrect = 0
    ninstance = 0
    nloss = 0
    (X, Y) = dtst
    for i = 1:bsize:size(X,2)
      x = X[:, i:i+bsize-1]
      ygold = Y[:, i:i+bsize-1]
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
