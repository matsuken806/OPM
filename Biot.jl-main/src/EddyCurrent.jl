"""
The function evaluates the equivalent circuit (eddy currents, voltages, etc) in a system of coupled PointPaths.
    The inputs are:
    PP_All which is an array of Nx3 matrices. Each of which are separate (magnetically coupled) loops of wire.
    CurrentInputList is a array of length(PP_All) is is the input currents to the system. So loops of wire you want to investigate the current in should be set to zero.
    f is the test frequency
    TestPoints is an Mx3 matrix to evaluate the field at

    Outputs: LMat, GMat_allFreq, CircOutputs_allFreq, CircInputs_allFreq, CircOutput_Key,RxInds,Φ
"""
function eval_EddyCurrent_PP_List(PP_All, CurrentInputList,f ,TestPoints = [0 0 0],OpenTurns = zeros(length(PP_All)); DownSampleFac=1,PlotOn=false,NPtsPath=100,NLayers=20,WireRadius=0.001,Quasistatic=true,NodeNodeCap = 1e-15)
    ρ_Cu = 1.72e-8
    WireArea = pi*WireRadius^2
    N = length(PP_All)
    LMat = zeros(N, N)
    SavedBSelfArr = [];
    Mut = 0.
    Self = 0.
    CurrentInputList = CurrentInputList[:]


    Mut, Self, SavedBSelfArr = Mutual_L_TwoLoops(PP_All[end], PP_All[1];DownSampleFac=DownSampleFac,MeasLayers=NLayers,MinThreshold=1e-10,WireRadius=WireRadius,IncludeWireInduct=true,SaveΦ₁=true)


    LMat[end,end] = Self

    start_time = time()  # Record the start time of the loop
    total_iterations = N * (N - 1) / 2  # Total number of iterations
    completed_iterations = 0  # Counter for completed iterations

    for j in 1:N, k in (j+1):N
        completed_iterations += 1
        if k == (j+1)
            println("Calculating Mutual Inductance between loop ", j, " and loop ", k, " (", completed_iterations, "/", total_iterations, ")")
            
            # Increment completed iterations
            
            
            # Calculate elapsed time and estimate remaining time
            elapsed_time = time() - start_time  # Time in seconds
            avg_time_per_iteration = elapsed_time / completed_iterations
            remaining_time = avg_time_per_iteration * (total_iterations - completed_iterations)
            
            # Convert remaining time to minutes and seconds
            remaining_minutes = Int(floor(remaining_time / 60))
            remaining_seconds = Int(round(remaining_time % 60))
            
            println("Estimated time remaining: ", remaining_minutes, " minutes and ", remaining_seconds, " seconds")
            
            Mut, Self, SavedBSelfArr = Mutual_L_TwoLoops(PP_All[j], PP_All[k]; DownSampleFac=DownSampleFac, MeasLayers=NLayers, MinThreshold=1e-10, WireRadius=WireRadius, IncludeWireInduct=true, SaveΦ₁=true)
        else
            Mut = Mutual_L_TwoLoops(PP_All[j], PP_All[k], SavedBSelfArr; DownSampleFac=DownSampleFac, MeasLayers=NLayers, MinThreshold=1e-10, WireRadius=WireRadius)
        end
        
        LMat[j, j] = Self
        LMat[j, k] = Mut
        LMat[k, j] = Mut
    end


    WireConductance = zeros(N,N)
    WireLength = zeros(N)
    eye = zeros(N,N)
    for jj in 1:N
    PP = vcat(PP_All[jj],PP_All[jj][1,:]')
    dL = ([VecDist(PP[I-1:I,:]) for I in 2:length(PP[:,1])])
    if OpenTurns[jj]==0 #if the turn is not left open
        WireLength[jj] =  sum([sqrt(dL[ii][1]^2+dL[ii][2]^2+dL[ii][3]^2) for ii in 1:length(dL)])
        WireConductance[jj,jj] = 1 / (WireLength[jj]*ρ_Cu/WireArea)

    else
        WireConductance[jj,jj] = 0. #if the turn is open, it does not conduct.
    end
    eye[jj,jj] = 1
    end
    N_f = length(f)

    N_in = sum(CurrentInputList.!=0)
    InputInds = findall((CurrentInputList.!=0)[:])
    CircOutput_Key =Array{String}(undef,(2*N+N_in))
    RxInds = [zeros(N_in)...,OpenTurns...,zeros(N)...].==1
    NotRxInds = [zeros(N_in)...,(OpenTurns.==0)...,zeros(N)...].==1
    CircOutput_Key[1:N_in] .= "Voltage on Current Source"
    CircOutput_Key[RxInds] .= "Voltage on Rx"
    CircOutput_Key[NotRxInds] .= "Voltage on Current-Carrying Wire"
    CircOutput_Key[1+N_in+N:end] .= "Current in Wire"


    GMat_allFreq = Complex.(zeros(2*N+N_in,2*N+N_in,N_f))
    CircOutputs_allFreq = Complex.(zeros(2*N+N_in,N_f))
    CircInputs_allFreq = Complex.(zeros(2*N+N_in,N_f))
    CircOutputs = []
    CircInputs = []

    Φ = Complex.(zeros(length(TestPoints[:,1]),3,N_f))
    for ff in 1:N_f
        freq = f[ff]
        λ = 3e8/freq
        TotalLength = 0.
        if !Quasistatic
            CurrentInputList = Complex.(CurrentInputList)
            for ii in 1:N_in
                TotalLength+=WireLength[InputInds[ii]]

                CurrentInputList[InputInds[ii]] *=exp(TotalLength/λ*2π*1im) 
            end
        end

        GMat = Complex.(zeros(2*N+N_in,2*N+N_in))
        GMat[1+N_in:N+N_in,1+N_in:N+N_in] = WireConductance
        GMat[(N_in+N+1):end,(N_in+N+1:end)] = -1 .* LMat .* 2 .*pi .* freq .* 1im
        GMat[1+N_in:N+N_in,N+1+N_in:end] =-1 .* eye
        GMat[N+N_in+1:end,1+N_in:N+N_in] =-1 .* eye
        for kk in 1:N_in
            I = Int(InputInds[kk]) #Current loop index

            GMat[kk,kk] = WireConductance[I,I]
            GMat[(N_in+I),(N_in+I)] = WireConductance[I,I]
            GMat[kk,(N_in+I)] = -1* WireConductance[I,I]
            GMat[(N_in+I),kk] = -1* WireConductance[I,I]
            # GMat[N_in+I,N_in+I] = 0
        end
        CapAdmittance = 2π*freq*NodeNodeCap*1im 
        # GMat[1:(N_in+N),1:(N_in+N)] .+= CapAdmittance
        GMat_allFreq[:,:,ff] = GMat
        CircInputs = Complex.(zeros(2*N+N_in,1))
        CircInputs[1:N_in] = CurrentInputList[InputInds]
        # println(CircInputs[:])
        # println(GMat)
        CircOutputs = pinv(GMat)*CircInputs[:]
        CircOutputs_allFreq[:,ff] = CircOutputs
        CircInputs_allFreq[:,ff] = CircInputs

        # println(sum(CircOutputs[RxInds]))
        # println(abs(CircOutputs[findfirst(RxInds)])/abs(sum(CircOutputs[RxInds])))
        # display(imshow(real.(GMat)))
        # println(size(GMat))

        Wires = vcat(PP_All...)
        #BiotSav(PointPath,dL,r,L) is the fast version
        for jj in 1:N
            PP = vcat(PP_All[jj],PP_All[jj][1,:]')
            dL = ([VecDist(PP[I-1:I,:]) for I in 2:length(PP[:,1])])
            # WireResist[jj] = sum([sqrt(dL[ii][1]^2+dL[ii][2]^2+dL[ii][3]^2) for ii in 1:length(dL)])*ρ_Cu/WireArea
            PP_RS = ([PP[I,:] for I in 1:length(PP[:,1])])
            L = length(PP_RS[:,1])    
            for i in 1:length(TestPoints[:,1]) 
            minDist = minimum(sqrt.(sum((Wires .- repeat(transpose(TestPoints[i,:]),length(Wires[:,1]),1)).^2,dims=2)))
                if minDist>= (0.5*WireRadius)
                    Φ[i,:,ff] .+=   (BiotSav(PP_RS,dL, TestPoints[i,:],L) .* CircOutputs[N_in+N+jj])[:] 
                else
                    println("Point too close to a wire")
                end
            end
        end
        ΦMag = [real.(sqrt(sum(Φ[ii,:,ff].^2))) for ii in 1:length(Φ[:,1,ff])]

        MeanMag = sum(ΦMag)/length(ΦMag)
        ΦMag[ΦMag.>(3*MeanMag)].=3*MeanMag
        if PlotOn
            pygui(true)
            plot(TestPoints[:,1],(ΦMag),TestPoints[:,2],(ΦMag),TestPoints[:,3],(ΦMag))
            # plot(Wires[1:100:end,1],Wires[1:100:end,2],"r*")
        end
    end


    return LMat, GMat_allFreq, CircOutputs_allFreq, CircInputs_allFreq, CircOutput_Key,RxInds,Φ

end




# GMat = Complex.(zeros(2*N+N_in,2*N+N_in))
# GMat[1+N_in:N+N_in,1+N_in:N+N_in] = WireConductance
# GMat[(N_in+N+1):end,(N_in+N+1:end)] = -1 .* LMat .* 2 .*pi .* freq .* 1im
# GMat[1+N_in:N+N_in,N+1+N_in:end] = eye
# GMat[N+N_in+1:end,1+N_in:N+N_in] = eye
# for kk in 1:N_in
#     I = Int(InputInds[kk]) #Current loop index

#     GMat[kk,kk] = WireConductance[I,I]
#     GMat[(N_in+I),(N_in+I)] = WireConductance[I,I]
#     GMat[kk,(N_in+I)] = -1* WireConductance[I,I]
#     GMat[(N_in+I),kk] = -1* WireConductance[I,I]
#     # GMat[N_in+I,N_in+I] = 0
# end