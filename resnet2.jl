using Knet

function add_conv!(params, height, width, input, output; init_mode=:output, bias=false)
   assert(init_mode in [:output, :input])
   w = randn(height, width, input, output) *
      sqrt(2.0 / (height * width * (init_mode==:output ? output : input)))
   push!(params, w)
   if bias
      b = zeros(1, 1, output, 1)
      push!(params, b)
   end
end

function add_linear!(params, input_size, output_size)
   w = randn(output_size, input_size) * sqrt(
      2.0 / (input_size * output_size))
   b = zeros(output_size, 1)
   push!(params, w)
   push!(params, b)
end

function add_bnorm!(params, stats, depth)
   push!(params, ones(1, 1, depth, 1))
   push!(params, zeros(1, 1, depth, 1))
   push!(stats, zeros(1, 1, depth, 1))
   push!(stats, zeros(1, 1, depth, 1))
end

function bnorm(w, s, x; mode=:train, momentum=.9, eps=1e-5)
   assert(mode in [:train, :test])
   if mode == :test
      x_hat =  (x .- s[1]) ./ (s[2] .+ eps)
      return w[1] .* x_hat .+ w[2]
   end
   m = size(x, 1) * size(x, 2) * size(x, 4)
   mu = sum(x, (1, 2, 4)) ./ m
   sigma2 = sumabs2(x, (1, 2, 4)) ./ m
   # println(mean(AutoGrad.getval(mu)))
   # println(mean(AutoGrad.getval(sigma2)))
   # println(size(s[1]))
   # println(size(mu))
   copy!(s[1], momentum .* s[1] .+ (1 - momentum).*AutoGrad.getval(mu))
   copy!(s[2], momentum .* s[2] .+ (1 - momentum).*AutoGrad.getval(sigma2))
   #println(s[1])
   x_hat = (x .- mu) ./ sqrt(sigma2 .+ eps)
   return w[1] .* x_hat .+ w[2]
end

function add_shortcut!(w, s, input, output)
   if input == output
      return
   end
   add_conv!(w, 1, 1, input, output)
   add_bnorm!(w, s, output)
end

function shortcut(w, s, x; mode=:train)
   if length(w) == 0
      return x
   end
   return bnorm(w[2:3], s, conv4(w[1], x; stride=2))
end

function add_basic_block!(w, s, input, output; wranges=nothing, sranges=nothing)
   start = length(w)+1
   sstart = length(s)+1
   add_conv!(w, 3, 3, input, output)
   add_bnorm!(w, s, output)
   add_conv!(w, 3, 3, output, output)
   add_bnorm!(w, s, output)
   add_shortcut!(w, s, input, output)
   if wranges !==nothing
      push!(wranges, start:length(w))
   end
   if sranges !==nothing
      push!(sranges, sstart:length(s))
   end
end

function basic_block(w, s, x; mode=:train)
   # println(size(w[1]), size(x))
   o1 = conv4(w[1], x; padding=1, stride=1+Int(size(w[1], 4) != size(w[1], 3)))
   # println("o1", size(o1))
   o2 = relu(bnorm(w[2:3], s[1:2], o1; mode=mode))
   o3 = conv4(w[4], o2; padding=1)
   o4 = bnorm(w[5:6], s[3:4], o3; mode=mode)
   return relu(o4 .+ shortcut(w[7:end], s[5:end], x; mode=mode))
end

function init_model(;n=3, dtype=Array{Float32})
   w = Any[]
   s = Any[]
   ranges = Any[]
   sranges = Any[]
   add_conv!(w, 3, 3, 3, 16)
   add_bnorm!(w, s, 16)
   for i = 1:n
      add_basic_block!(w, s, 16, 16; wranges=ranges, sranges=sranges)
   end
   add_basic_block!(w, s, 16, 32; wranges=ranges, sranges=sranges)
   for i = 1:n-1
      add_basic_block!(w, s, 32, 32; wranges=ranges, sranges=sranges)
   end
   add_basic_block!(w, s, 32, 64; wranges=ranges, sranges=sranges)
   for i = 1:n-1
      add_basic_block!(w, s, 64, 64; wranges=ranges, sranges=sranges)
   end
   add_linear!(w, 64, 10)
   w = map(x->convert(dtype, x), w)
   s = map(x->convert(dtype, x), s)
   w, s, ranges, sranges
end

#=
   Requires s and ranges comes from the upper scope
=#
function resnet(w, x; mode=:train)
   # println(size(x))
   o = relu(bnorm(w[2:3], s[1:2], conv4(w[1], x; padding=1); mode=mode))
   # println(size(o))
   for i = 1:3n
      #=println(i)
      println(wranges[i])
      println(sranges[i])=#
      o = basic_block(w[wranges[i]], s[sranges[i]], o; mode=mode)
   end
   return w[end-1] * mat(pool(o; mode=2, window=(8,8))) .+ w[end]
end

function loss(w,x,ygold)
    scores = resnet(w, x)
    #=penalty = 0.0
    for i = 1:2:length(w)
      penalty += sum(w[i].^2)
   end=#
    return -sum(ygold .* logp(scores, 1)) ./ size(ygold, 2)  #+ decay * penalty
end

function loaddata(;nval=5000, augment=true, dtype=Array{Float32})
   include("data.jl")
   include("imgproc.jl")
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

function next_batch(x, y, bs; augmented=true, dtype=Array{Float32})
   batch_indices = rand(1:size(x, 4), bs)
   x, y =  x[:, :, :, batch_indices], y[:, batch_indices]
   if augmented
      x_ = convert(typeof(x), zeros(32, 32, size(x,3), size(x,4)))
      for i = 1:bs
         rstart = rand(1:5)
         cstart = rand(1:5)
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

function train(w, dtrn, dtst; dtype=Array{Float32}, iters=15000, bsize=32, print_period=1000, lr=0.1, momentum=0.9, augmented=true, actual_trn=nothing, decay=1e-5)
   #weights, predict, loss = model(;decay=decay)
   #model = (weights, predict, loss)
   report(iter)=println((:iter,iter,:trn,accuracy(w,actual_trn; dtype=dtype),:val,accuracy(w,dtst; dtype=dtype)))
   #report(iter)=println((:iter,iter, :val,accuracy(w,dtst)))
    #w = weights()
    prms = map(x->Momentum(lr=lr, gamma=momentum), w)
    
    for i = 1:iters
        x, y = next_batch(dtrn[1], dtrn[2], bsize; augmented=augmented, dtype=dtype)
        # println(size(x), " ", size(y))
        # println(Array{Float32}(sum(y, 1)))
        g = lossgrad(w, x, y)
        println("Num grads ", length(g))   
        for j = 1:length(prms)
            #println(sum(g[j] ./ length(g[j])))
            update!(w[j], g[j], prms[j])
        end
      println("iter ", i)
      println(" ")
      if i % print_period == 0
         report(i)
      end
   end
   return w
end

function accuracy(w,dtst; dtype=Array{Float32})
    ncorrect = 0
    ninstance = 0
    nloss = 0
    nloss_count = 0
    X, Y = dtst
    
    for i = 1:100:size(Y,2)
      x = convert(dtype, X[:, :, :, i:i+99])
        ygold = convert(dtype, Y[:, i:i+99])
        
      ypred = resnet(w, x)#; mode=:test)
      nloss += loss(w, x, ygold)
      ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
      ninstance += size(ygold, 2)
      nloss_count += 1
      #=println("iter acc ", i, " ", ncorrect / ninstance)
      println(" ")=#
    end
    println(ncorrect, " ", ninstance," ", nloss, " ", nloss_count)
   return (ncorrect / ninstance, nloss / nloss_count)
end

# Model specs
n = 3
dtype = KnetArray{Float32}
dtrn, dtrn_, dval, dtst = loaddata()
w, s, wranges, sranges = init_model(;dtype=dtype)
lossgrad = grad(loss)
train(w, dtrn_, dval; actual_trn=dtrn, dtype=dtype, bsize=64, iters=30000)
