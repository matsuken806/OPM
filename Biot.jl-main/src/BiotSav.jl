VecDist(X::Array) = [ X[2,1]-X[1,1], X[2,2]-X[1,2] , X[2,3]-X[1,3]]
#X is the 2 coordinates to take distance between in format of [x₁,y₁,z₁;x₂,y₂,z₂]

"""
This function takes in:
    "Point Path" which is an Nx3 set of coordinates that define a current loop
    "r" which is a 1x3 vector that is a point in space to evaluate the 3D B field at

    kwargs:
        "current" is a scalar multiplier. Default is 1 Amp 
        "MinThreshold" is a minimum distance that the "r" point must be from any wire
            The function tends to inf near wires

"""
function BiotSav(PointPath,r;Current=1 ,MinThreshold = 0.001)
    
    PointPath = vcat(PointPath,PointPath[1,:]')
    NPts = length(PointPath[:,1])
    # dB = zeros(NPts-1,3)
    dB = [0. , 0., 0.]
    minDist = minimum(sqrt.(sum((PointPath .- repeat(transpose(r[:]),NPts,1)).^2,dims=2)))
    if minDist>= MinThreshold
        for I in 2:NPts
            dL = VecDist(PointPath[I-1:I,:])
            MeanPoint = sum(PointPath[I-1:I,:],dims=1)[:] ./ 2
            Rprime = r .- MeanPoint

            RDist = sqrt(sum(Rprime.^2))
            R̂ = Rprime/RDist

                if Current isa AbstractArray
                    # dB[I-1,:] = μ₀/(4*π) .* Current[I].*LinearAlgebra.cross(dL,R̂[:]) ./ (RDist)^2
                    dB .+= 1e-7 .* Current[I].*LinearAlgebra.cross(dL,R̂[:]) ./ (RDist)^2
                
                else
                    # dB[I-1,:] = μ₀/(4*π) .* Current.*LinearAlgebra.cross(dL,R̂[:]) ./ (RDist)^2
                    dB .+= 1e-7 .* Current.*LinearAlgebra.cross(dL,R̂[:]) ./ (RDist)^2
                end
                

        end
    # else
    #    plot(r[1],r[2],"r*")
    end
    return dB
end
"""
"Point Path" which is an Nx3 set of coordinates that define a current loop
    "r" which is a 1x3 matrix that is a point in space to evaluate the 3D B field at
    dL is the vector distance between points in the point path
    L is the length of PointPath

"""
function BiotSav(PointPath::Matrix,dL,r::Vector,L::Int;MinThreshold=1e-5)
    
    dB = [0. , 0., 0.]
    
    MeanPoint = (PointPath[2:end,:] .+ PointPath[1:end-1,:])./2
    if MinThreshold !== nothing
    minDist = minimum(sqrt.(sum((PointPath .- repeat(transpose(r[:]),size(PointPath,1),1)).^2,dims=2)))
        if minDist>= MinThreshold
            for I in 2:L
                # Rprime = r .- PointPath[I]
                # 
                # MeanPoint = sum(PointPath[I-1:I,:],dims=1)[:] ./ 2
                
                Rprime = r .- MeanPoint[I-1,:][:]
                RDist = (sum(Rprime.^2))
                dB .+= 1e-7 .*LinearAlgebra.cross(dL[I-1],(Rprime/sqrt(RDist))[:]) ./ (RDist)
            end
        end
    else 
         for I in 2:L
            # Rprime = r .- PointPath[I]
            # 
            # MeanPoint = sum(PointPath[I-1:I,:],dims=1)[:] ./ 2
            Rprime = r .- MeanPoint[I-1,:][:]
            RDist = (sum(Rprime.^2))
            dB .+= 1e-7 .*LinearAlgebra.cross(dL[I-1],(Rprime/sqrt(RDist))[:]) ./ (RDist)
        end
        
    end

    
       
    
    dB
end

"""
Evaluate the Biot-Savart equation at point r given the wire defined by PointPath.
\n
Inputs: PointPath is a vector of vectors

"""
function BiotSav(PointPath::Vector,dL,r::Vector,L::Int;MinThreshold=1e-5)
    

    dB = [0. , 0., 0.]
    
    MeanPoint = (PointPath[2:end] .+ PointPath[1:end-1])./2

    
    if MinThreshold !== nothing
        minDist = minimum([sqrt.(sum((PointPath[i] .- r[:]).^2)) for i in 1:L])
            if minDist>= MinThreshold
                for I in 2:L
                    # Rprime = r .- PointPath[I]
                    # 
                    # MeanPoint = sum(PointPath[I-1:I,:],dims=1)[:] ./ 2
                    Rprime = r .- MeanPoint[I-1]
                    RDist = (sum(Rprime.^2))
                    dB .+= 1e-7 .*LinearAlgebra.cross(dL[I-1],(Rprime/sqrt(RDist))[:]) ./ (RDist)
                end
            end
        else 
            for I in 2:L
                # Rprime = r .- PointPath[I]
                # 
                # MeanPoint = sum(PointPath[I-1:I,:],dims=1)[:] ./ 2
                Rprime = r .- MeanPoint[I-1]
                RDist = (sum(Rprime.^2))
                dB .+= 1e-7 .*LinearAlgebra.cross(dL[I-1],(Rprime/sqrt(RDist))[:]) ./ (RDist)
            end
            
        end
    
    dB
end

function BiotSav(PointPath,dL,r::Matrix,L::Int;MinThreshold=1e-5)
    

    BiotSav(PointPath,dL,r[:],L;MinThreshold=MinThreshold)
end

function BiotSav(PointPath,r::Matrix)
    

    BiotSav(PointPath,r[:])
end

