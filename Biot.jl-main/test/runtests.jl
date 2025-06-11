using Biot
using Test

@testset "Biot.jl" begin
    # Write your tests here.
    println("Running tests for Biot.jl...")

    # Test BiotSav
    Lmat, Lmat_sum = eval_Induct_CircToroid(0.005, 0.0010, 20; PlotOn=true, WireRadius = 0.0005)
    println("Inductance Matrix Lmat: ", Lmat)
    println("Sum of Inductance Matrix Lmat_sum: ", Lmat_sum)
end
