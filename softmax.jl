module Softmax
   using Knet
   dtype = Float32
   use_gpu = false
   family = :linear

   function init_params(ninputs, noutputs, winit=0.0001)
      W = winit * randn(dtype, ninputs, noutputs)
      b = zeros(dtype, noutputs, 1)
      return [W, b]
   end

   function predict(w,x)
      w[1]' * x .+ w[2]
   end

   function loss(w,x,ygold; lambda=1e-5)
       scores = predict(w, x)
       # println("logp", size(logp(scores, 1)))
       # println("loss", size(ygold))
       penalty = dtype(0)
       for i = 1:2:length(w)
         penalty = lambda * sum(w[i] .^ 2)
      end
       return -sum(ygold .* logp(scores, 1)) ./ size(x, 2) + penalty
   end

   lossgradient =  grad(loss)# your code here [just 1 line]

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
end
