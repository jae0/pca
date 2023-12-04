
using MultivariateStats, Statistics
using RDatasets: dataset

using Turing
using LinearAlgebra

# Packages for visualization
using VegaLite, DataFrames, StatsPlots

# Import Fisher's iris example data set
using RDatasets

using Random
Random.seed!(1111);



showall(x) = show(stdout, "text/plain", x)


function discretize_decimal( x, delta=0.01 ) 
  num_digits = Int(ceil( log10(1.0 / delta)) )   # time floating point rounding
  out = round.( round.( x ./ delta; digits=0 ) .* delta; digits=num_digits)
  return out
end


