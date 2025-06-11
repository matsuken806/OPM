# include("FieldCalc.jl")
# # VecDist(X::Array) = [ X[2,1]-X[1,1], X[2,2]-X[1,2] , X[2,3]-X[1,3]]
# #X is the 2 coordinates to take distance between in format of [x₁,y₁,z₁;x₂,y₂,z₂]

"""
This function takes in an inner and outer radius and makes the ideal "D-shaped" toroid core

It returns a list of coordinates (r,z) where r is the radial location
on the toroid, and z is the height from the center axis. In a cross-section
view this is essentially x and y.


r₁ - inner radius
r₂ - outer radius
dr (kwarg) is the dr between each sampled location


"""
function DCoreGeom(r₁, r₂; dr = 0.0001,NPts = nothing,PlotOn=false,UpsamplePoints = 1e4)
    dZdr(r) = log.(sqrt(r₁ * r₂) ./ r) ./ sqrt.(log.(r ./ r₁) .* log.(r₂ ./ r))
    if ~(NPts===nothing)
        dr = (r₂ - r₁) / UpsamplePoints
    end
    rᵢ = r₁ + dr / 10#starting at a small fraction above zero, to avoid an inf.
    r = [rᵢ]
    zᵢ = 0.0
    z = [zᵢ] # initializing vector
    II = 1 #iterator
    NumIters = floor((r₂ - r₁) / dr)

    while (NumIters > II)
        dz = dZdr(rᵢ)
        zᵢ +=dz*dr
        rᵢ += dr

        push!(z, zᵢ)
        push!(r, rᵢ)
        II += 1
    end
    z .-= z[end] #the boundary condition at the end must be fixed to equal 0
    
    z[1] = 0 #because of the inf initial dz/dr, the initial boundary must be manually set to zero
    ZFlatHeight = z[2]
    StartPts = Int(ceil(ZFlatHeight/dr))
    for i in 1:StartPts
        insert!(z,1+i,z[1+i]*i/StartPts)
        insert!(r,1+i,r[1])
    end

    z = vcat(z[:],-1 .*reverse(z[2:end-2]))
    r = vcat(r[:],reverse(r[2:end-2]))
    DownsampleIndex = 1:Int(round(UpsamplePoints/NPts)):length(z)
    z = z[DownsampleIndex]
    r = r[DownsampleIndex]
    if PlotOn
        plot(r, z)
    end
    return r, z
end


function DCore_PointPath(r₁, r₂; NPts = 100)
    r, z = DCoreGeom(r₁, r₂; NPts = NPts*10,PlotOn=false)
    CumulativeDist = 0
    for i in 1:(length(r)-1)
        CumulativeDist += sqrt((r[i+1]-r[i])^2+(z[i+1]-z[i])^2)
    end
    TargetDist = CumulativeDist/(NPts+1)
    rᵣ = [r[1]]
    zᵣ = [z[1]]
    CumulativeDist2 = 0
    for i in 1:(length(r)-1)
        CumulativeDist2 += sqrt((r[i+1]-r[i])^2+(z[i+1]-z[i])^2)
        if CumulativeDist2>TargetDist
            push!(rᵣ,r[i])
            push!(zᵣ,z[i])
            CumulativeDist2 = 0
        end
    end
    

    PointPathZeros = zeros(size(rᵣ))
    PointPath = hcat(rᵣ,zᵣ,PointPathZeros)
    return PointPath
end

function FieldMapDToroid(r₁, r₂; NPts = 100, TestLayers = 20)

    PointPath = DCore_PointPath(r₁, r₂;NPts = NPts)
    FieldMapPointPath(PointPath, TestLayers; WeightRadius = true, InvWeights = true)
end



"""
RectCoords=[ 1   2.5 0
         5   2.5 0
         5  -2.5 0
         1  -2.5 0]

         Dr₁ = 1
         Dr₂ = 5

         Cr₁ = 2
         Cr₂ = 2
         CircCenter = [3,0,0]
         CompareD_Circ_Rect_Toroid(Dr₁, Dr₂,RectCoords,Cr₁, Cr₂,CircCenter,20)

"""
function CompareD_Circ_Rect_Toroid(Dr₁, Dr₂,RectCoords,Cr₁, Cr₂,CircCenter,TestLayers = 20)
    figure(75)
    FieldMapDToroid(Dr₁,Dr₂)


    figure(76)
    R_PointPath = MakeRectPointPath(RectCoords;NElement = 50, PlotOn=false)
    FieldMapPointPath(R_PointPath, 20; WeightRadius = true,InvWeights = true)

     figure(77)
    C_PointPath = MakeEllip(Cr₁,Cr₂;Center = CircCenter,NPts=100)
    C_PointPath = reverse(C_PointPath,dims=1)
FieldMapPointPath(C_PointPath, 20; WeightRadius = true,InvWeights = true)

end
