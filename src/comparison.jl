using CSV, JSON, MixedModels, NLopt, StatsModels

const LNopts = [:LN_BOBYQA, :LN_COBYLA, :LN_NELDERMEAD, :LN_PRAXIS, :LN_SBPLX]

struct Results
    dnm::String
    model::String
    opts::Vector{Symbol}
    cvgtyp::Vector{Symbol}
    nfeval::Vector{Int}
    minobj::Vector{Float64}
    secs::Vector{Float64}
end

function doexample(ddir, dnm, model, optimizers::Vector{Symbol}=LNopts)
    dat = CSV.read(joinpath(ddir, string(dnm, ".csv")))
    for (n, v) in eachcol(dat)
        if !any(ismissing, v)
            dat[n] = disallowmissing(v)
        end
    end
    rhsexpr = Meta.parse(model)
    m = fit(LinearMixedModel, @eval(@formula(Y ~ $rhsexpr)), dat)
    cvgtyp = similar(optimizers)
    nfeval = similar(optimizers, Int)
    minobj = similar(optimizers, Float64)
    secs = similar(optimizers, Float64)
    optsum = m.optsum
    for i in eachindex(optimizers)
        optsum.optimizer = optimizers[i]
        secs[i] = @elapsed(refit!(m))
        cvgtyp[i] = optsum.returnvalue
        nfeval[i] = optsum.feval
        minobj[i] = optsum.fmin
    end
    Results(dnm, model, optimizers, cvgtyp, nfeval, minobj, secs)
end
