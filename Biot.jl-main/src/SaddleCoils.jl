
"""
Makes a point path for the BiotSav function
    Input is an Nx3 array of coordinates where it is structured as:
    [x₁, y₁, z₁
    ..
    ..
    xₙ, yₙ, zₙ]

Coords = [1 0 0
          2 0 0
          2 1 0
          1 1 0]
"""
function MakeRectPointPath(Coords;NElement = 50, PlotOn=false)
    Coords = vcat(Coords, reshape(Coords[1,:],1,3))
    xVec = Coords[:,1]
    yVec = Coords[:,2]
    zVec = Coords[:,3]
    xVecUp = [Coords[1,1]]
    yVecUp = [Coords[1,2]]
    zVecUp = [Coords[1,3]]
    for i in 1:(length(xVec)-1)
        UpSamp = LinRange(xVec[i],xVec[i+1],NElement)[:]
        xVecUp = vcat(xVecUp,UpSamp,Coords[i+1,1])
        UpSampy = LinRange(yVec[i],yVec[i+1],NElement)[:]
        yVecUp = vcat(yVecUp,UpSampy,Coords[i+1,2])
        UpSampz = LinRange(zVec[i],zVec[i+1],NElement)[:]
        zVecUp = vcat(zVecUp,UpSampz,Coords[i+1,3])
    end
    PointPath = unique(hcat(xVecUp,yVecUp,zVecUp);dims=1)
    if PlotOn
        pygui(true)
        scatter3D(xVecUp,yVecUp,zVecUp)
    end
    return PointPath

end



"""
Makes a point path for the BiotSav function
    Input is an Nx3 array of coordinates where it is structured as:
    [θ₁, r₁, z₁
    ..
    ..
    θₙ, rₙ, zₙ]

    Angle must be in degrees!

    It then outputs CARTESIAN coordinates for the PP

Coords = [1 0 0
          2 0 0
          2 1 0
          1 1 0]
"""
function MakeRectSaddlePointPath(Coords;NElement = 50, PlotOn=false,ConnectFirstLastPoints = true)
    if ConnectFirstLastPoints
        Coords = vcat(Coords, reshape(Coords[1,:],1,3))
    end
    θVec = Coords[:,1]
    rVec = Coords[:,2]
    zVec = Coords[:,3]
    θVecUp = [Coords[1,1]]
    rVecUp = [Coords[1,2]]
    zVecUp = [Coords[1,3]]
    for i in 1:(length(θVec)-1)
        UpSamp = LinRange(θVec[i],θVec[i+1],NElement)[:]
        θVecUp = vcat(θVecUp,UpSamp,Coords[i+1,1])
        UpSampr = LinRange(rVec[i],rVec[i+1],NElement)[:]
        rVecUp = vcat(rVecUp,UpSampr,Coords[i+1,2])
        UpSampz = LinRange(zVec[i],zVec[i+1],NElement)[:]
        zVecUp = vcat(zVecUp,UpSampz,Coords[i+1,3])
    end
    PointPathPolar = unique(hcat(θVecUp,rVecUp,zVecUp);dims=1)

    PointPath = similar(PointPathPolar)


        PointPath[:,1] = PointPathPolar[:,2] .* cosd.(PointPathPolar[:,1])
        PointPath[:,2] = PointPathPolar[:,2] .* sind.(PointPathPolar[:,1])
        PointPath[:,3] = PointPathPolar[:,3]
    if PlotOn
        pygui(true)
        scatter3D(PointPath[:,1],PointPath[:,2],PointPath[:,3])
    end
    return PointPath

end

MakePolarRectPointPath = MakeRectSaddlePointPath #Simply adding another name for the function to make it easier to find


function FieldSaddleCentered(h,D,ϕ)
    μ₀ = 4*pi*1e-7
    s = 1+ (h/D)^2

    B₀ = 4/pi * μ₀*(h/D^2)*(s^(-1/2)+ s^(-3/2))*sin(ϕ/2)
    return B₀
end