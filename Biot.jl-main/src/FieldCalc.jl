## Biot savart law
# using LinearAlgebra
# using PyPlot
# using MPI_Tools
# using Interpolations


"""
Makes a point path of an ellipse with r₁,r₂ as two radii
normal direction is assumed to be in Z
kwargs: Center is the center of path [X,Y,Z]
NPts is the number of points along path


"""
function MakeEllip(r₁,r₂;Center = [0,0,0],NPts=100)
    X,Y,Z = ([r₁ .* cos.(I/NPts *2*π) for I in 1:NPts].+Center[1],
             [r₂ .* sin.(I/NPts *2*π) for I in 1:NPts].+Center[2],
             [0 for I in 1:NPts].+Center[3])
    
    Coords = hcat(X,Y,Z)

end


"""
Simple analytical formula for the field on axis of a Circle
    R - radius in meters
    z - axial offset in meters
    
    kwargs:
        I = current in coil

"""
function FieldOnAxis_Circ(R,z;I=1)
    μ₀ = 4*pi*1e-7
    Bz = μ₀*2*pi*R^2*I ./ (4*π* (z^2 + R^2)^(3/2))
end

# x = -.3:.01:.3
# y = x
# DefField = [BiotSav(MakeEllip(1,1;Center = [0,0,0],NPts=1000),[i,j,k];MinThreshold= 0.05)[3]
#             for i in x,
#                 j in y,
#                 k=0]
# X,Y = MPI_Tools.meshgrid(x,y)
# PyPlot.surf(X[:],Y[:],DefField[:],cmap="jet")

function MakeEllipTestPoints(r₁,r₂;Center = [0,0,0],NPts=6,Layers = 3)
    LL=1
    X,Y,Z = ([r₁*LL/(Layers+1) .* cos.(I/NPts *2*π) for I in 1:NPts].+Center[1],
             [r₂ *LL/(Layers+1) .* sin.(I/NPts *2*π) for I in 1:NPts].+Center[2],
             [0 for I in 1:NPts].+Center[3])
             Coords = hcat(X,Y,Z)
    for LL in 2:Layers
        X₁,Y₁,Z₁ = ([r₁*LL/(Layers+1) .* cos.(I/NPts *2*π) for I in 1:NPts].+Center[1],
                 [r₂ *LL/(Layers+1) .* sin.(I/NPts *2*π) for I in 1:NPts].+Center[2],
                 [0 for I in 1:NPts].+Center[3])
                 Coords = hcat(X,Y,Z)
        X = vcat(X,X₁)
        Y = vcat(Y,Y₁)
        Z = vcat(Z,Z₁)

     end
     return X,Y,Z
 end


"""
This function takes in a pointpath [X,Y,Z] array and maps the interior

Inputs are
PointPath [X,Y,Z] , N x 3 array
NumLayers - number of interior test points

"""
 function FieldMapPointPath(PointPath, NumLayers; WeightRadius = false,InvWeights = false, PlotAxes = nothing)

     xTP = PointPath[:,1]
     x   = PointPath[:,1]
     yTP = PointPath[:,2]
     y   = PointPath[:,2]
     x₁ = minimum(xTP)
     x₂ = maximum(xTP)
     y₁ = minimum(yTP)
     y₂ = maximum(yTP)
     xmid = (x₁+ x₂)/2
     ymid = (y₁+ y₂)/2



     for i in 1:(NumLayers-2)
         xTP = vcat(xTP,(x .- xmid) .* (i/NumLayers) .+ xmid)
         yTP = vcat(yTP,(y .- ymid) .* (i/NumLayers) .+ ymid)
     end
     TPList = hcat(xTP,yTP,zeros(size(yTP)))
     TP_Arr =[([TPList[i,1],TPList[i,2],TPList[i,3]]) for i in 1:length(xTP)]
     if WeightRadius
         Weights = PointPath[:,1]
     else
          Weights = ones(size( PointPath[:,1]))
      end
      if InvWeights
          Weights = 1 ./Weights
      end
      push!(Weights,Weights[1])
     BMag = [BiotSav(PointPath,TP_Arr[i];Current = Weights,MinThreshold =1e-8)[3] for i in 1:length(TP_Arr)]

     BMag = abs.(BMag)

     BPlot = findall(xx-> xx!=0,BMag)
     if PlotAxes!== nothing
        PyPlot.axes(PlotAxes)
     end

     surf(xTP[BPlot],yTP[BPlot],BMag[BPlot],cmap="jet")


 end



function WireNearField_B(r,Wire_R;I=1)
    μ₀ = 4*π*1e-7
    if r<Wire_R
        B = μ₀*r*I / (2*π*Wire_R^2)
    elseif r==Wire_R
        B = μ₀*I / (2*π*Wire_R)
    else
        B = μ₀*I / (2*π*r)
    end
    return B
end

function DerivField(r,Wire_R=0.001;I=1)
    μ₀ = 4*π*1e-7
    if r<Wire_R
        B = 0 ##Update later.
    elseif r==Wire_R
        B=0
    else
        B = -1*μ₀*I / (2*π*r^2)
    end
    return B
end

