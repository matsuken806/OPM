module Biot

using LinearAlgebra, PyPlot, Interpolations

μ₀ = 4*pi*1e-7


include("FieldCalc.jl")
include("BiotSav.jl")
include("Toroid_Inductance.jl")
include("MakeSolenoid.jl")
include("ToroidOptimizer.jl")
include("EddyCurrent.jl")
include("Transformer.jl")
include("MutualInductanceLoops.jl")
include("SaddleCoils.jl")

export  BiotSav,
FieldMapPointPath,MakeEllipTestPoints,FieldOnAxis_Circ,MakeEllip,eval_EddyCurrent_PP_List,
MakeEllipSolenoidPointPath,makeConcentricSolenoids,EvalField_Centered,MakeRectSaddlePointPath,MakePolarRectPointPath,
eval_Induct_DToroid,eval_Induct_CircToroid,eval_Induct_DToroidTransformer,eval_Induct_DToroid_SplitWindings,
Mutual_L_TwoLoops,eval_LMat_PP_List,LMatMutualIndexing

# Write your package code here.

end
