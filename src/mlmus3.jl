using CSV, DataFrames, FileIO, StatFiles

const dtadir = "/tmp/mlmus3/"

function dodir(dnm)
    for fn in filter(r"\.dta$", readdir(dnm))
        dofile(dnm, fn)
    end
end
function dofile(dnm, fn)
    dd = DataFrame(load(joinpath(dnm, fn)))
    for (n,v) in eachcol(dd)
        if isa(v, Vector{Float64})
            try
                dd[n] = convert(Vector{Int}, v)
            end
        elseif isa(v, Vector{Union{Float64, Missing}})
            try
                dd[n] = convert(Vector{Union{Int,Missing}}, v)
            end
        elseif isa(v, Vector{Union{Float32, Missing}})
            try
                dd[n] = convert(Vector{Union{Int32, Missing}}, v)
            end
        end
    end
    CSV.write(string(splitext(fn)[1], ".csv"), dd)
end

const LETTERS = string.('A':'Z')
const letters = string.('a':'z')
const digits = string.('0':'9')
const Letters = vcat(LETTERS, letters)
const digitsLetters = vcat(digits, Letters)
const ldouble = vec([i*j for j in digitsLetters, i in Letters])
const ltriple = vec([i*j for j in ldouble, i in Letters])

function toLabels!(df::DataFrame, cnms::Vector{Symbol})
    for nm in cnms
        catcol = droplevels!(categorical(df[nm]))
        nlev = length(levels(catcol))
        if nlev ≤ length(LETTERS)
            df[nm] = LETTERS[catcol.refs]
        elseif nlev ≤ length(Letters)
            df[nm] = Letters[catcol.refs]
        elseif nlev ≤ length(ldouble)
            df[nm] = ldouble[catcol.refs]
        elseif nlev ≤ length(ltriple)
            df[nm] = ltriple[catcol.refs]
        end
    end
    df
end

function toNY!(df::DataFrame, cnms::Vector{Symbol})
    for nm in cnms
        col = df[nm]
        if all(x -> iszero(x) || x == 1, col)
            df[nm] = ["N","Y"][df[nm] .+ 1]
        else
            catcol = droplevels!(categorical(df[nm]))
            if length(levels(catcol)) == 2
                df[nm] = ["N","Y"][catcol.refs]
            end
        end
    end
    df
end
