

"""
This function makes an elliptic solenoid and calculates the inductance.
    The function calculates the flux through the center of each loop's cross-section
        Lₛ = Φ/I (self inductance = flux per amp)
        Lₘ₂₁ = (Φ₂₁ ⋅ Φ₂₂/|Φ₂₂|) /I
            -mutual inductance from the 1ˢᵗ wire on 2ⁿᵈ is flux from 1 in the 2ⁿᵈ's cross section
            that is aligned with the flux from the second on itself. All per ampere
        Lₘ₂₁ = Lₘ₁₂
        Total L = ∑ᵢ∑ⱼ(Lᵢⱼ)
    The inputs are:
    N - Number of turns
    r₁ - one radius of the ellipse
    r₂ - Second radius of ellipse
    L - length of solenoid (axially)
    kwargs:
    MeasLayers - Number of concentric layers to calc field
                    The field is calculated over concentric ellipses (linearly spaced)
                    This is the number of concentric test point layers
    NLayers -  Number of test points per layer
    NPtsPath  - number of points that each wire loop will be discretized into,
    PlotOn deternimes if the coil will be plotted as it builds

"""
function eval_Solenoid_Induct(
    N, #Number of turns
    r₁, #Radius 1 of ellipse
    r₂,#Radius 2 of ellipse
    L; #Length of solenoid in meters
    DownSampleFac=1,PlotOn=false,NPtsPath=100,NLayers=20)

    if PlotOn
        pygui(true)
        gcf()
    end
    PointPath_SingleLoop =  MakeEllip(r₁, r₂;NPts=NPtsPath)
    
    LMat = zeros(N, N)
    if PlotOn
        scatter3D(PointPath_SingleLoop[:,1], PointPath_SingleLoop[:,2], PointPath_SingleLoop[:,3])
    end
    NewPP = MakeEllip(r₁,r₂;Center = [0, 0, 1/ N * L],NPts = NPtsPath)

    if PlotOn
        scatter3D(NewPP[:,1], NewPP[:,2], NewPP[:,3])
    end

    Mut, Self, SavedBSelfArr = Mutual_L_TwoLoops(PointPath_SingleLoop, NewPP;DownSampleFac=DownSampleFac,MeasLayers=NLayers,MinThreshold=1e-10,IncludeWireInduct=false,SaveΦ₁=true)
    LMat .+= Self .* OffDiagOnes(N, 1)
    LMat .+= Mut .* OffDiagOnes(N, 2)
    for i in 3:N
        NewPP = MakeEllip(r₁,r₂;Center = [0, 0, (i-1)/ N * L],NPts = NPtsPath)

        if PlotOn
            scatter3D(NewPP[:,1], NewPP[:,2], NewPP[:,3])
        end
        Mut = Mutual_L_TwoLoops(PointPath_SingleLoop, NewPP, SavedBSelfArr;DownSampleFac=DownSampleFac,MeasLayers=NLayers,MinThreshold=1e-10)
        println(Mut)
        
        LMat .+= Mut .* OffDiagOnes(N, i)
        
        println(i)
    end
    LMat[N,N] = LMat[1,1]
    return LMat, sum(LMat)


end

"""
This function makes an elliptic solenoid and calculates the field at the center of the solenoid
        Lₘ₂₁ = Lₘ₁₂
        Total L = ∑ᵢ∑ⱼ(Lᵢⱼ)
    The inputs are:
    N - Number of turns
    r₁ - one radius of the ellipse
    r₂ - Second radius of ellipse
    L - length of solenoid (axially)
    kwargs:

    NPts_Coil  - number of points that each wire loop will be discretized into,
"""
function EvalField_Centered(
    N, #Number of turns
    r₁, #Radius 1 of ellipse
    r₂,#Radius 2 of ellipse
    L; #Length of solenoid in meters
    NPts_Coil=100
    )



            FieldCentered = [
                BiotSav(
                    MakeEllip(
                                r₁, #Radius 1 of ellipse
                                r₂;#Radius 2 of ellipse
                                Center = [0, 0, (Wᵢ - 1) / N * L], #The solenoid's axis is in Z
                                NPts = NPts_Coil, #How discretized the windings are
                                ),
                        [0, 0, L/2];
                        MinThreshold = 0.001,
                        )[3]
                        for Wᵢ in 1:N
                        ]

    return sum(FieldCentered)
end

"""
This function takes in a vector for each of the following:
    Number of turns (NVec) in each of the coupled solenoids
    r₁/₂ are the first and second radii of the elliptic cross section
    L is the length of each solenoid section
    Zoffset is the center offset distance of each solenoid. 

    The solenoids are built symmetrically so the Z offset is the center location
    
"""
function makeConcentricSolenoids(NVec,r₁Vec,r₂Vec,LVec,ZOffsetVec;
    DownSampleFac=1,PlotOn=false,NPtsPath=50,NLayers=15)

    NCoils = length(NVec)
    PP_List = vcat([[MakeEllipSolenoidPointPath(NVec[i],r₁Vec[i],r₂Vec[i],LVec[i],ZOffsetVec[i];NPtsPath=NPtsPath)] for i in 1:NCoils]...)
    LMat = zeros(NCoils,NCoils)

    if PlotOn
        for j in 1:NCoils
            pygui(true)
            plot3D(PP_List[j][:,1],PP_List[j][:,2],PP_List[j][:,3])
        end
    end

    for j in 1:NCoils
        OtherCoilsIndex = filter(x-> x!==j,1:NCoils) #Make a vector of ind of other coils

        for k in OtherCoilsIndex
            
            Mut, Self, SavedBSelfArr = Mutual_L_TwoLoops(PP_List[j], PP_List[k];DownSampleFac=DownSampleFac,MeasLayers=NLayers,MinThreshold=1e-10,IncludeWireInduct=false,SaveΦ₁=false)
            LMat[j,j] = Self
            LMat[j,k] = Mut
            LMat[k,j] = Mut
        end

    end
    

    return LMat,PP_List
end


function MakeEllipSolenoidPointPath(N,r₁,r₂,FullLength,Zoffset;
    NPtsPath=100)
    dL = FullLength / (N-1)  
    DriveCoilPointPath = vcat([MakeEllip(r₁, r₂; Center = [0, 0, L+Zoffset], NPts = NPtsPath) for L in (-FullLength/2:dL:FullLength/2)]...)  
    return DriveCoilPointPath
end