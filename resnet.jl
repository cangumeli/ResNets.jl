module ResNet

include("data.jl")
include("imgproc.jl")
using Knet, AutoGrad

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
    stdv = 1 ./ sqrt(input_size)
    uniform(dims) = rand(dims) * 2stdv - stdv
    w = uniform((output_size, input_size))
    b = uniform((output_size, 1))
    push!(params, w)
    push!(params, b)
end

function add_bnorm!(params, stats, depth)
    push!(params, ones(1, 1, depth, 1))
    push!(params, zeros(1, 1, depth, 1))
    push!(stats, zeros(1, 1, depth, 1))
    push!(stats, ones(1, 1, depth, 1))
end

function bnorm(w, x, s, rng; mode=:train, momentum=.9, eps=1e-9, freezed=false, scnt=nothing)
    assert(mode in [:train, :test])
    assert(length(rng) == 2)
    if mode == :test
        x_hat = (x .- s[rng[1]]) ./ sqrt(s[rng[2]] .+ eps)
        return w[1] .* x_hat .+ w[2]
    end
    m = size(x, 1) * size(x, 2) * size(x, 4)
    mu = sum(x, (1, 2, 4)) ./ m
    x_mu = x .- mu
    sigma2 = sumabs2(x_mu, (1, 2, 4)) ./ m
    x_hat = x_mu ./ sqrt(sigma2 .+ eps)
    if freezed && scnt != nothing
        s[rng[1]] = (scnt[rng[1]]*s[rng[1]] + AutoGrad.getval(mu)) / (scnt[rng[1]] + 1)
        s[rng[2]] = ((scnt[rng[2]]+1)*s[rng[2]] + AutoGrad.getval(sigma2)) / (scnt[rng[2]] + 2)
        scnt[rng[1]] += 1
        scnt[rng[2]] += 1
    else # the standard running average mode
        s[rng[1]] = momentum * s[rng[1]] + (1 - momentum) * AutoGrad.getval(mu)
        s[rng[2]] = momentum * s[rng[2]] + (1 - momentum) * AutoGrad.getval(sigma2)
    end
    return w[1] .* x_hat .+ w[2]
end

function add_shortcut!(w, s, input, output; use_conv=true)
    if input == output
        return
    end
    if use_conv
        add_conv!(w, 1, 1, input, output)
        add_bnorm!(w, s, output)
    end
end

#= Requires dtype to come from the upper scope =#
function shortcut(w, x, s, rng; bottleneck=false, mode=:train, use_conv=true, freezed=false, scnt=nothing)
    if length(w) == 0
        if bottleneck && !use_conv
            x_mat = mat(x)
            x_padded = vcat(x_mat, dtype(zeros(size(x_mat))))
            x_ = reshape(x_padded, (size(x)[1:2]..., size(x, 3) * 2, size(x,4)))
            return pool(x_)
        end
        return x
    end
    if use_conv
        return bnorm(w[2:3], conv4(w[1], x; stride=2), s, rng[1:2]; mode=mode, freezed=freezed, scnt=scnt)
    end
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

function add_basic_block_pre!(w, s, input, output; wranges=nothing, sranges=nothing)
    start = length(w)+1
    sstart = length(s)+1
    add_bnorm!(w, s, input)
    add_conv!(w, 3, 3, input, output)
    add_bnorm!(w, output)
    add_conv!(w, 3, 3, output, output)
    add_shortcut!(w, s, input, output)
    if wranges !==nothing
        push!(wranges, start:length(w))
    end
    if sranges !==nothing
        push!(sranges, sstart:length(s))
    end
end

function basic_block(w, x, s, rng; mode=:train, freezed=false, scnt=nothing)
    o1 = conv4(w[1], x; padding=1, stride=1+Int(size(w[1], 4) != size(w[1], 3)))
    o2 = relu(bnorm(w[2:3], o1, s, rng[1:2]; mode=mode, freezed=freezed, scnt=scnt))
    o3 = conv4(w[4], o2; padding=1)
    o4 = bnorm(w[5:6], o3, s, rng[3:4]; mode=mode, freezed=freezed, scnt=scnt)
    #println(" ")
    #println("o4 ", size(o4))
    #println("x ", size(x))
    o0 = shortcut(w[7:end], x, s, rng[5:end]; bottleneck=size(o4, 3)!=size(x, 3), mode=mode, freezed=freezed, scnt=scnt)
    #println(size(o4), " ", size(o0))
    return relu(o4 .+ o0)
end

function basic_block_pre(w, x, s, rng; mode=:train)
    o1 = relu(bnorm(w[1:2], x, s, rng[1:2]; mode=mode))
    o2 = conv4(w[3], x; padding=1, stride=1+Int(size(w[1], 4) != size(w[1], 3)))
    o3 = relu(bnorm(w[4:5], x, s, rng[3:4]; mode=mode))
    o4 = conv4(w[6], x, s; padding=1)
    o0 = shortcut(w[7:end], x, s, rng[5:end]; bottleneck=size(o4, 3)!=size(x, 3), mode=mode)
    return relu(o4 + o0)
end

function add_bottleneck_block_pre!(w, s, input, output; wranges=nothing, sranges=nothing)
    start = length(w) + 1
    sstart = length(s) + 1
    nbottleneck = output / 4
    add_bnorm!(w, s, input)
    add_conv!(w, 1, 1, input, nbottleneck)
    add_bnorm!(w, s, nbottleneck)
    add_conv!(w, s, 3, 3, nbottleneck, nbottleneck)
    add_bnorm!(w, s, nbottleneck)
    add_conv!(w, s, 1, 1, nbottleneck, output)
    add_shortcut!(w, s, input, output)
    if wranges !==nothing
        push!(wranges, start:length(w))
    end
    if sranges !==nothing
        push!(sranges, sstart:length(s))
    end
end

function bottleneck_block_pre(w, x, s, rng; mode=:train)
    o1 = relu(bnorm(w[1:2], x, s, rng[1:2]; mode=mode))
    o2 = conv4(w[3], o1)
    o3 = relu(bnorm(w[4:5], o2, s, rng[3:4]; mode=mode))
    o4 = conv4(w[6], o3; padding=1, stride=1+Int(length(w) >= 10))
    o5 = relu(bnorm(w[7:8], o4, s, rng[5:6]; mode=mode))
    o6 = conv4(w[9], o5)
    o0 = shortcut(w[10:end], x, s, rng[7:end]; mode=mode)
    return o6 + o0
end

function init_model(;n=3, pre=false, add_block! = add_basic_block!)
    w = Any[]
    s = Any[]
    ranges = Any[]
    sranges = Any[]
    add_conv!(w, 3, 3, 3, 16)
    if !pre
        add_bnorm!(w, s, 16)
    end
    for i = 1:n
        add_block!(w, s, 16, 16; wranges=ranges, sranges=sranges)
    end
    add_block!(w, s, 16, 32; wranges=ranges, sranges=sranges)
    for i = 1:n-1
        add_block!(w, s, 32, 32; wranges=ranges, sranges=sranges)
    end
    add_block!(w, s, 32, 64; wranges=ranges, sranges=sranges)
    for i = 1:n-1
        add_block!(w, s, 64, 64; wranges=ranges, sranges=sranges)
    end
    if pre
        add_bnorm!(w, s, 64)
    end
    add_linear!(w, 64, 10)
    w = map(x->convert(dtype, x), w)
    s = map(x->convert(dtype, x), s)
    w, s, ranges, sranges, map(x->0, s)
end


#=
Requires s, n, wranges and sranges to come from the upper scope
=#
function resnet(w, x; mode=:train, freezed=false, scnt=nothing)
    # block = basic_block
    o = relu(bnorm(w[2:3], conv4(w[1], x; padding=1), s, 1:2; mode=mode, freezed=freezed, scnt=scnt))
    for i = 1:3n
        o = basic_block(w[wranges[i]], o, s, sranges[i]; mode=mode, freezed=freezed, scnt=scnt)
    end
    return w[end-1] * mat(pool(o; mode=2, window=(8,8))) .+ w[end]
end


function resnet_pre(w, x; mode=:train, freezed=false, scnt=nothing, block=bottleneck_block_pre)
    o = conv4(w[1], x; padding=1)
    for i = 1:3n
        o = block(w[wranges[i]], o, s, sranges[i]; mode=mode, freezed=freezed, scnt=scnt)
    end
    fm = relu(bnorm(w[(end-3):(end-2)], o, s, (length(s)-1):length(s), freezed=freezed, scnt=scnt))
    return w[end-1] * mat(pool(fm; mode=2, window=(8,8))) .+ w[end]
end


# function resnet_pre(w, x; mode=:train, block=bottlenect
function result_loss(w, scores, ygold)
    penalty = 0.0
    for i = 1:length(w)
        if size(w[i], ndims(w[i])) != 1
            # penalty += sum(0.5lambda .* (w[i] .* w[i]))
            penalty += sum(0.5lambda .* (w[i] .* w[i]))
        end
    end
    return -sum(ygold .* logp(scores, 1)) ./ size(ygold, 2)  + penalty
end

function loss(w,x,ygold; mode=:train)
    scores = resnet(w, x; mode=mode)
    penalty = 0.0
    for i = 1:length(w)
        if size(w[i], ndims(w[i])) != 1
            penalty += sum(0.5lambda .* (w[i] .* w[i]))
        end
    end
    return -sum(ygold .* logp(scores, 1)) ./ size(ygold, 2) + penalty
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

    #mnt = imgproc.mean_subtract!(xtrn;mode=:pixel)
    mnt = mean(xtrn, (3, 4))
    xtrn .-= mnt
    xtst .-= mnt
    if nval > 0
        xval .-= mnt
    end
    #=mnt = mean(xtrn, 4)
    xtst .-= mnt
    xtrn .-= mnt
    xval .-= mnt=#
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

function next_batch(x, y, bs; augmented=true)
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

function train(w, dtrn, dtst; s=nothing, iters=15000, bsize=32, print_period=1000, lr=0.1,
               momentum=0.9, augmented=true, actual_trn=nothing, decay=1e-4, reset_cnt=true, lr_decay_iter=(32000, 48000),
               bnorm_iters=5000, bi_print_period=250, val_only=false)
    global dtype
    println("Using dtype ", dtype)
    report = nothing
    if val_only
        report = iter->println((:iter,iter,:trn,:val,accuracy(w,dtst; bmode=:test)))
    else
        report = iter->println((:iter,iter,:trn,accuracy(w,actual_trn; bmode=:test),:val,accuracy(w,dtst; bmode=:test)))
    end
    if n >= 18
        lr *= .1
    end
    prms = map(x->Momentum(lr=lr, gamma=momentum), w)
    #=if reset_cnt
    global_state[:nforward] = 0
    end=#
    report(0)
    global lossgrad
    for i = 1:iters
        x, y = next_batch(dtrn[1], dtrn[2], bsize; augmented=augmented)
        g = lossgrad(w, x, y)
        for j = 1:length(prms)
            update!(w[j], g[j], prms[j])
        end
        if i == 400 && n >= 18
            for j = 1:length(prms)
                prms[j].lr *= 10
            end
        end
        if i in lr_decay_iter
            for j = 1:length(prms)
                prms[j].lr *= 0.1
            end
        end
        if i % 50 == 0 || i == 1
            #println("stats ", map(x->(sum(x)/length(x)), s))
            println("iter ", i)
            println(" ")
        end
        if i % print_period == 0
            report(i)
        end
    end
    if bnorm_iters > 0
        reset_stats(s; dtype=dtype)
        scnt = map(x->0, s)
        for i = 1:bnorm_iters
            x, _ = next_batch(dtrn[1], dtrn[2], bsize; augmented=augmented)
            resnet(w, x; mode=:train, freezed=true, scnt=scnt)
            if i % bi_print_period == 0
                report(i)
            end
            #println(scnt)
        end
    end
    return w
end

function reset_stats(s)
    for i = 1:length(s)
        s[i] = (i%2 == 0) ? dtype(ones(size(s[i]))) : dtype(zeros(size(s[i])))
    end
end

function accuracy(w, dtst; bmode=:train)
    
    println("Computing Accuracy...")
    ncorrect = 0
    ninstance = 0
    nloss = 0
    nloss_count = 0
    X, Y = dtst
    if bmode == :train
        reset_stats(s; dtype=dtype)
    end
    for i = 1:100:size(Y,2)
        x = convert(dtype, X[:, :, :, i:i+99])
        ygold = convert(dtype, Y[:, i:i+99])
        #=println(typeof(x))
        println(typeof(y))=#
        if i % 1000 == 0
            println("Accuracy iter ", i)
        end
        ypred = resnet(w, x; mode=bmode)
        nloss += result_loss(w, ypred, ygold) # diminish the side effects
        ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
        ninstance += size(ygold, 2)
        nloss_count += 1
    end
    println(ncorrect, " ", ninstance," ", nloss, " ", nloss_count)
    return (ncorrect / ninstance, nloss / nloss_count)
end

global n = 3
global dtype = KnetArray{Float64}
global lambda = .0001
global nval = 5000
global bsize = 128
# Model specs
function init(_n=3, _dtype=nothing, _nval=5000, _bsize=128, _lambda=.0001)
    global n = _n
    global dtype = _dtype == nothing ? (Knet.gpu() >= 0 ? KnetArray{Float32} : Array{Float32}) : _dtype
    global lambda = _lambda
    global nval = _nval
    global bsize = _bsize
    w_, s_, wranges_, sranges_, _ = init_model(;n=_n)
    global w = w_
    global s = s_
    global wranges = wranges_
    global sranges = sranges_
    global lossgrad = getgrad()
end

function getgrad()
    return grad(loss)
end

function train()
    dtrn, dtrn_, dval, dtst = loaddata(;augment=true, nval=nval)
    low_power = Knet.gpu() < 0
    global w, lambda, s, sranges, wranges
    # TODO: make options configurable
    w = train(w, dtrn_, nval > 0 ? dval : dtst; s=s, actual_trn=dtrn, bsize=bsize,
              iters=64000, print_period=500, augmented=true, lr_decay_iter=(32000, 48000),
              val_only=low_power, bnorm_iters=3000)
    println("Final accuracy", accuracy(w, dtst; dtype=dtype, bmode=:test))
end

end
