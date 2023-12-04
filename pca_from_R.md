# Example work flow from R calling pca functions

Following is based upon examples here: 

https://cran.r-project.org/web/packages/JuliaCall/readme/README.html

https://hwborchers.github.io/


```R

if (0) {
    # to set up:
    # julia_setup(JULIA_HOME = "the folder that contains julia binary")
    # options(JULIA_HOME = "the folder that contains julia binary")
    # Set JULIA_HOME in command line environment.
    install.packages("JuliaCall")
    install_julia()
}

# load JuliaCall interface
library(JuliaCall)
julia <- julia_setup()

# set up paths: adjust to local copy of github.com/jae0/pca 
project_directory = file.path("~", "bio", "pca"  )

# load pca functions
# julia_source cannot traverse directories .. temporarily switch directory
currwd = getwd() 
    setwd(project_directory) 
    julia_source( "startup.jl" ) 
    julia_source( "pca_functions.jl" ) 
setwd(currwd) # revert


# Example data: replicate iris analysis
IR = scale(iris[, 1:4])
IR.id = iris$Species

X = t(IR)  # pca operates upon transpose of data

julia_assign("X", X)  # copy data into julia session

J = julia_command  # copy to shorten following text calls

# set up problem parameters and conduct basic PCA to get initial settings
J( "nvar, nobs = size(X)" )
J( "nq = 2; n_samples = 1000" ) # nq is number of (latent) factors 
J( "U, sigma_pca, C, PC = pca_standard(X')" ) # return values: eigenvectors, sqrt(eigenvalues), correlation matrix, pc scores
J( "v = eigenvector_to_householder(U, nq)" )  # v is the householder representation of the eigenvectors used by the householder pca

# param sequence = sigma_noise, sigma(nq), v, r=norm(v)~ 1.0 (scaled)
J( "init_params = [0.1; sigma_pca[1:nq]; v; 1.0 ]" )
J( "Mph = PCA_BH_model(X, nq )" )  # all dims == default form
J( "chain = sample(Mph, NUTS(), n_samples; init_params=init_params)" )
J( "sigma = PCA_BH_extract( chain, X, nq, :sigma )" )  # sqrt(eigenvalues)
J( "scores = PCA_BH_extract( chain, X, nq, :scores )" ) 
J( "eigenvectors = PCA_BH_extract( chain, X, nq, :eigenvectors )" )
 
# to move data into R
sigma = julia_eval("sigma")  # standard deviations
scores = julia_eval("scores")  # pc scores
eigenvectors = julia_eval("eigenvectors") # weights

# save as it seems JuliaCall alters plotting environment
fn = file.path("~", "tmp", "pca_scores_posteriors.Rdata" )
saveRDS( scores, file=fn )

# in an alternate R-session, load results and plot
fn = file.path("~", "tmp", "pca_scores_posteriors.Rdata" )
scores = readRDS(fn)
sp1 = 1:50
sp2 = 51:100
sp3 =101:150
plot( scores[2,,] ~ scores[1,,], type="n" )
points( scores[2,sp1,] ~ scores[1,sp1,], col="red", pch="."  )
points( scores[2,sp2,] ~ scores[1,sp2,], col="blue", pch="."  )
points( scores[2,sp3,] ~ scores[1,sp3,], col="grey", pch="."  )

```


