_build = false
for p in ("AutoGrad","ArgParse","JLD","Knet")
    if Pkg.installed(p) == nothing
        Pkg.add(p)
        _build = true
    end
end
if _build
    Pkg.build("Knet")
end
#TODO: add argument parsing stuff
resnet = include("resnet3.jl")
resnet.init(3)
resnet.train(;low_power=true)
