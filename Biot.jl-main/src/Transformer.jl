"""
The function evaluates the equivalent circuit (eddy currents, voltages, etc) in a system of coupled PointPaths.
    The inputs are:
    PP_All which is an array of Nx3 matrices. Each of which are separate (magnetically coupled) loops of wire.
    CurrentInputList is a array of length(PP_All) is is the input currents to the system. So loops of wire you want to investigate the current in should be set to zero.
    f is the test frequency
    TestPoints is an Mx3 matrix to evaluate the field at
"""
function eval_LMat_PP_List(PP_All; DownSampleFac=1,PlotOn=false,NPtsPath=100,NLayers=20,WireRadius=0.001)
    ρ_Cu = 1.72e-8
    WireArea = pi*WireRadius^2
    N = length(PP_All)
    LMat = zeros(N, N)
    SavedBSelfArr = [];
    Mut = 0.
    Self = 0.
    PP_All, NewIndexes = SortPPByArea(PP_All;MeasLayers=NLayers,WireRad = WireRadius)
    println("Sorted")

    Mut, Self, SavedBSelfArr,TestPoints, Weights = Mutual_L_TwoLoops(PP_All[end], PP_All[1];DownSampleFac=DownSampleFac,MeasLayers=NLayers,MinThreshold=1e-10,WireRadius=WireRadius,IncludeWireInduct=false,SaveΦ₁=true)


    LMat[end,end] = Self
    
    start_time = time()  # Record the start time of the loop
    total_iterations = N * (N - 1) / 2  # Total number of iterations
    completed_iterations = 0  # Counter for completed iterations

    for j in 1:N, k in (j+1):N
        completed_iterations += 1
        if k==(j+1)
            # Calculate elapsed time and estimate remaining time
            elapsed_time = time() - start_time  # Time in seconds
            avg_time_per_iteration = elapsed_time / completed_iterations
            remaining_time = avg_time_per_iteration * (total_iterations - completed_iterations)
            
            # Convert remaining time to minutes and seconds
            remaining_minutes = Int(floor(remaining_time / 60))
            remaining_seconds = Int(round(remaining_time % 60))
            
            println("Estimated time remaining: ", remaining_minutes, " minutes and ", remaining_seconds, " seconds")

            Mut, Self, SavedBSelfArr,TestPoints, Weights= Mutual_L_TwoLoops(PP_All[j], PP_All[k];DownSampleFac=DownSampleFac,MeasLayers=NLayers,MinThreshold=1e-10,WireRadius=WireRadius,IncludeWireInduct=false,SaveΦ₁=true)
        else
            Mut = Mutual_L_TwoLoops(PP_All[j], PP_All[k], SavedBSelfArr,TestPoints, Weights;DownSampleFac=DownSampleFac,MeasLayers=NLayers,MinThreshold=1e-10,WireRadius=WireRadius)
        end
        # println(j)
        

        LMat[j,j] = Self
        LMat[j,k] = Mut
        LMat[k,j] = Mut
    end

    


    return LMat,NewIndexes
end

function getPPArea(PP;MeasLayers=55,WireRad = 0.2e-3)

    TestPoints, Weights = Biot.PP_2_TestPoints_arbPlane(PP, MeasLayers)
    SliceArea = sum(Weights)
    return SliceArea 
end

function SortPPByArea(PP_All;MeasLayers=55,WireRad = 0.2e-3)
    N = length(PP_All)
    Areas = zeros(N)
    for k in 1:N
        Areas[k] = getPPArea(PP_All[k];MeasLayers=MeasLayers,WireRad = WireRad)
    end
    AreasMat = hcat(Areas,1:N)
    AreasSort = sortslices(AreasMat;dims=1)
    PP_AllNew = similar(PP_All)
    NewIndexes = Int.(AreasSort[:,2])
    for i=1:N
        PP_AllNew[i] = PP_All[NewIndexes[i]]
    end
    return PP_AllNew, NewIndexes
end

