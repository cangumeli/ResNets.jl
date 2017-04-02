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
   return params
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

function add_resnet_block!(w, s, depth)
   add_conv!(w, 3, 3, depth, depth)
   add_bnorm!(w, s, depth)
   add_conv!(w, 3, 3, depth, depth)
   add_bnorm!(w, s, depth)
end

function add_resnet_block_with_bottleneck!(w, s, input_depth, output_depth)
   add_conv!(w, 1, 1, input_depth, output_depth)
   add_conv!(w, 3, 3, input_depth, output_depth)
   add_bnorm!(w, s, output_depth)
   add_conv!(w, 3, 3, output_depth, output_depth)
   add_bnorm!(w, s, output_depth)
end


function init_resnet_model(n; dtype=Array{Float32})
   w = Any[]
   s = Any[]
   add_conv!(w, 3, 3, 3, 16)
   add_bnorm!(w, s, 16)
   for i = 1:n
      add_resnet_block!(w, s, 16)
   end
   add_resnet_block_with_bottleneck!(w, s, 16, 32)
   for i = 1:(n-1)
      add_resnet_block!(w, s, 32)
   end
   add_resnet_block_with_bottleneck!(w, s, 32, 64)
   for i = 1:(n-1)
      add_resnet_block!(w, s, 64)
   end
   add_linear!(w, 64, 10)
   w = map(x->convert(dtype, x), w)
   s = map(x->convert(dtype, x), s)
   return w, s
end

function bnorm(w, s, x; mode=:train, eps=1e6, momentum=0.9)
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
   copy!(s[1], momentum .* s[1] .+ (1 - momentum).*AutoGrad.getval(mu))
   copy!(s[2], momentum .* s[2] .+ (1 - momentum).*AutoGrad.getval(sigma2))
   #println(s[1])
   x_hat = (x .- mu) ./ sqrt(sigma2 .+ eps)
   return w[1] .* x_hat .+ w[2]
end

function conv_bnorm_relu(w, s, x; mode=:train, padding=1, stride=1)
   assert(length(w) == 3)
   assert(length(s) == 2)
   o1 = conv4(w[1], x; padding=padding, stride=stride)
   # println(size(s[1]))
   # println(size(s[2]))
   # println(size(o1))
   o2 = bnorm(w[2:3], s, o1; mode=mode)
   return relu(o2)
end

function resnet_block_forward(w, s, x; mode=:train)
   o0 = x
   # println(length(w))
   # println("Weight sizes...", map(x->size(x), w))
   bottleneck = length(w) == 7
   start = 1
   if bottleneck
      # println("Here")
      o0 = conv4(w[1], x; stride=2)
      start = 2
   end
   o1 = conv_bnorm_relu(w[start:(start+2)], s[1:2], x; mode=mode, stride=1+Int(bottleneck))
   # println("size o1 ", size(o1))
   o2 = conv_bnorm_relu(w[(start+3):(start+5)], s[3:4], o1; mode=mode)
   return o0 .+ o2
end

function resnet_forward(w, x, s, n; mode=:train)
   wstart = 4
   sstart = 3
   o = conv_bnorm_relu(w[1:3], s[1:2], x; mode=mode)
   # println("Passed the first step..")
   for i = 1:3n
      bottleneck = Int((i > n) & ((i % n) == 1))
      # println("bottleneck ", bottleneck)
      wend = wstart+5+bottleneck
      send = sstart+3
      # println("i ", i)
      o = resnet_block_forward(w[wstart:wend], s[sstart:send], o; mode=mode)
      wstart = wend + 1
      sstart = send + 1
   end
   # Global max pooling
   op = pool(o; mode=2, window=(8, 8))
   return w[end-1] * mat(op) .+ w[end]
end

function resnet_model(n; dtype=Array{Float32}, get_state=false)
   state = Dict{Any, Any}()
   function weights()
      w, state[:s] = init_resnet_model(n; dtype=dtype)
      return w
   end

   function predict(w, x; mode=:train)
      return resnet_forward(w, x, state[:s], n; mode=mode)
   end

   function loss(w,x,ygold)
      scores = predict(w, x)
      return -sum(ygold .* logp(scores, 1)) ./ size(x, 4)
   end

   if get_state
      return weights, predict, loss, state
   end
   return weights, predict, loss
end

# TODO: test the backprop
function test()
   weights, predict, loss, state = resnet_model(3; get_state=true)
   w = weights()
   for i = 1:length(w)
      println(size(w[i]))
   end
   lossgrad = grad(loss)
   x = rand(Float32, 32, 32, 3, 64)
   y = rand(Float32, 10, 64)
   #for i = 1:10
   lossgrad(w, x, y)
   #end
   # println(state[:s])
   #for i = 1:1000
      #println("iter ", i)
      #println("")
   #end
end
