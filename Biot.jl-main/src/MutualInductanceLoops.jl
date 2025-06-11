
"""
This function takes in two point paths and simulates the self inductance of the first, and mutual inductance between them.

    kwargs:
    DownSampleFac: this is the amount the point path will be down-sampled  to make the ``Test points''
        It is more accurate to have a highly sampled point path and less finely sampled interior. 
    
    
    MeasLayers is the number of interior test point layers
    WireRadius (slightly) changes the way test points are found. It mostly just adds a set of test points near the wire if the distance is large between the outer layer of test points and wire edge
    IncludeWireInduct is a boolean. If true, adds the wire's internal inductance to the self-inductance of the first
    MinThreshold should not be modified to be larger than the wire radius. It essentially makes the bior savart function ignore test points under that treshold, which can be useful to prevent divisions by zero in edge cases.

    

"""
function Mutual_L_TwoLoops(PP₁,PP₂;DownSampleFac=1,
    MeasLayers=55,# Number of concentric layers to calc field
    MinThreshold=0.000001,
    WireRadius=0.001,
    IncludeWireInduct=false,
    SaveΦ₁=false
    )

    TestPoints, Weights = PP_2_TestPoints_arbPlane(PP₁, MeasLayers, WireRadius;DownSampleFac=DownSampleFac)
    Weights .+= eps(typeof(Weights[1]))
    PtsPerLayer = length(PP₁[:,1])
    NPtsPerSlice = MeasLayers * PtsPerLayer  
    SliceArea = sum(Weights) 
        WeightsNorm = Weights / sum(Weights) * NPtsPerSlice  # normalizing
        if SaveΦ₁
            B_Self = zeros(length(TestPoints[:,1]), 3)
            Φ₁₁ = 0.
        else
            Φ₁₁ = 0.
        end
    
    Φ₂₁ = 0.
    for i in 1:length(TestPoints[:,1]) 
         
        if SaveΦ₁
            B_Self[i,:] =  BiotSav(PP₁, TestPoints[i,:];MinThreshold) .* WeightsNorm[i] .* SliceArea ./ NPtsPerSlice
            Φ₁₁ += √(sum(B_Self[i,:].^2))
            Φ₂₁ += sum(B_Self[i,:]  .* BiotSav(PP₂, TestPoints[i,:];MinThreshold) .* WeightsNorm[i] .* SliceArea ./ NPtsPerSlice) /    √(sum(B_Self[i,:].^2))
        else
            B_Self =  BiotSav(PP₁, TestPoints[i,:];MinThreshold) .* WeightsNorm[i] .* SliceArea ./ NPtsPerSlice
            Φ₁₁ += √(sum(B_Self.^2))
            
            Φ₂₁ += sum(B_Self .* BiotSav(PP₂, TestPoints[i,:];MinThreshold) .* WeightsNorm[i] .* SliceArea ./ NPtsPerSlice) /    √(sum(B_Self.^2))
        end
        
       
    end

    CumulativeDist = 0
    if IncludeWireInduct
        for i in 1:(length(PP₁[:,1]) - 1)
            CumulativeDist += sqrt(sum((PP₁[i,:] .- PP₁[i + 1,:]).^2))
        end
        # println(CumulativeDist)
        WireInductance = internalInductance(CumulativeDist)
        Φ₁₁ += WireInductance
    end

    return Φ₂₁, Φ₁₁, B_Self,TestPoints, Weights

end


function Mutual_L_TwoLoops(PP₁,PP₂,PriorBSelf;DownSampleFac=1,
    MeasLayers=55,# Number of concentric layers to calc field
    MinThreshold=0.000001,
    WireRadius=0.001,
   
    )

    TestPoints, Weights = PP_2_TestPoints_arbPlane(PP₁, MeasLayers, WireRadius;DownSampleFac=DownSampleFac)
    PtsPerLayer = length(PP₁[:,1])
    NPtsPerSlice = MeasLayers * PtsPerLayer  
    SliceArea = sum(Weights) 
        Weights = Weights / sum(Weights) * NPtsPerSlice  # normalizing
    
    Φ₂₁ = 0.


    PP₂ = vcat(PP₂,PP₂[1,:]')
    dL = ([VecDist(PP₂[I-1:I,:]) for I in 2:length(PP₂[:,1])])
    # PP₂_RS = ([PP₂[I,:] for I in 1:length(PP₂[:,1])])
    L = length(dL[:,1])+1    
    #BiotSav(PointPath,dL,r,L) is the fast version
    for i in 1:length(TestPoints[:,1]) 
        Φ₂₁ += sum(PriorBSelf[i,:] .* BiotSav(PP₂,dL, TestPoints[i,:],L) .* Weights[i] .* SliceArea ./ NPtsPerSlice) /    √(sum(PriorBSelf[i,:].^2))
        # Φ₂₁ += sum(PriorBSelf[i,:] .* BiotSav(PP₂, TestPoints[i,:]) .* Weights[i] .* SliceArea ./ NPtsPerSlice) /    √(sum(PriorBSelf[i,:].^2))
    end

    

    return Φ₂₁

end


function Mutual_L_TwoLoops(PP₁,PP₂,PriorBSelf,TestPoints, Weights;DownSampleFac=1,
    MeasLayers=55,# Number of concentric layers to calc field
    MinThreshold=0.000001,
    WireRadius=0.001,
   
    )

    # TestPoints, Weights = PP_2_TestPoints_arbPlane(PP₁, MeasLayers, WireRadius;DownSampleFac=DownSampleFac)
    PtsPerLayer = length(PP₁[:,1])
    NPtsPerSlice = MeasLayers * PtsPerLayer  
    SliceArea = sum(Weights) 
        WeightsNorm = Weights / sum(Weights) * NPtsPerSlice  # normalizing
    
    Φ₂₁ = 0.


    PP₂ = vcat(PP₂,PP₂[1,:]')
    dL = ([VecDist(PP₂[I-1:I,:]) for I in 2:length(PP₂[:,1])])
    # PP₂_RS = ([PP₂[I,:] for I in 1:length(PP₂[:,1])])
    L = length(dL[:,1])+1    

    #BiotSav(PointPath,dL,r,L) is the fast version
    for i in 1:length(TestPoints[:,1]) 
        Φ₂₁ += sum(PriorBSelf[i,:] .* BiotSav(PP₂,dL, TestPoints[i,:],L) .* WeightsNorm[i] .* SliceArea ./ NPtsPerSlice) /    √(sum(PriorBSelf[i,:].^2))
        # Φ₂₁ += sum(PriorBSelf[i,:] .* BiotSav(PP₂, TestPoints[i,:]) .* Weights[i] .* SliceArea ./ NPtsPerSlice) /    √(sum(PriorBSelf[i,:].^2))
    end

    

    return Φ₂₁

end