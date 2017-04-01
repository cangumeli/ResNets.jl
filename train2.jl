for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

include("data.jl")
include("imgproc.jl")
model = include("softmax.jl")
using Knet

function main()
    dtrn, dval, dtst = loaddata()
    w = model.init_params(size(dtrn[1], 1), size(dtrn[2],1))
    train(w, dtrn, dval; num_iters=1000)
    println("Final accuracy ", model.accuracy(w, dtst)[1])
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
   if model.family !== :conv
      xtrn = mat(xtrn)
      xval = mat(xval)
      xtst = mat(xtst)
   end
   return (xtrn, ytrn), (xval, yval), (xtst, ytst)
end

function init_optim(params; momentum=model.dtype(0.9), lr=model.dtype(.01))
   return map(x->Momentum(lr=lr, gamma=momentum), params)
end

function next_batch(dtrn; batchsize=128)
   (x, y) = dtrn
   num_examples = size(x,2)
   sample = rand(UInt32, batchsize) % num_examples + 1
   mbatch = x[:, sample], y[:, sample]
   if model.use_gpu
      mbatch = map(x->convert(KnetArray{model.dtype}, x), mbatch)
   end
   return mbatch
end

function train(w, dtrn, dval;
      optim_schedule=nothing, lr=0.01, num_iters=10000, print_period=50)
   println("Iter/Training accuracy: ",0, "/", model.accuracy(w, dtrn)[1])
   println("Iter/Test accuracy: ",0, "/", model.accuracy(w, dval)[1])
   println("")
   opt = init_optim(w; lr=lr)
   for i = 1:num_iters
      x, y = next_batch(dtrn)
      dw = model.lossgradient(w, x, y)
      if optim_schedule !== nothing
         optim_schedule(opt, i)
      end
      for j = 1:length(dw)
         update!(w[j], dw[j], opt[j])
      end
      if i % print_period == 0
         println("Iter/Training accuracy: ",i, "/", model.accuracy(w, dtrn)[1])
         println("Iter/Val accuracy: ",i, "/", model.accuracy(w, dval)[1])
         println("")
      end
   end
    return w
end

main()
