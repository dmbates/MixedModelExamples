using CSV, DataFrames, MixedModels

const LNopts = [:LN_BOBYQA, :LN_COBYLA, :LN_NELDERMEAD, :LN_PRAXIS, :LN_SBPLX]

function dospec(io, ddir, spec, optimizers::Vector{Symbol}=LNopts)
    cd(ddir) do
        for rw in eachrow(spec)
            dat = CSV.read(string(rw[:data], ".csv"))
            for (n, v) in eachcol(dat)
                if !any(ismissing, v)
                    dat[n] = disallowmissing(v)
                end
            end
            m = LinearMixedModel(@eval(@formula(Y ~ $(Meta.parse(rw[:model])))), dat)
            optsum = m.optsum
            optsum.maxfeval = 100_000
            for opt in optimizers
                println(rw[:id], ", ", opt)
                optsum.optimizer = opt
                secs = @elapsed(refit!(m))
                join(io, [rw[:id], opt, optsum.returnvalue, optsum.feval, optsum.fmin, secs], ",")
                println(io)
            end
        end
    end
end

function wrtspec(io, data, model)
    m = LinearMixedModel(@eval(@formula(Y ~ $(Meta.parse(model)))), CSV.read(joinpath("data", data*".csv"), allowmissing=:none))
    trms = m.trms
    n, p = size(trms[end-1])
    q = sum(x -> size(x, 2), trms) - (p + 1)
    join(io, [n, p, q, length(getθ(m)), length(trms) - 2], ",")
end

function crtspec(inpt, outpt)
    ff = readlines(inpt)
    open(outpt, "w") do io
        println(io, "id,data,model,n,p,q,ngrp,ntheta")
        for j in 2:length(ff)
            print(io, ff[j], ',')
            id, data, model = split(ff[j], ',')
            wrtspec(io, data, model)
            println(io)
        end
    end
end
