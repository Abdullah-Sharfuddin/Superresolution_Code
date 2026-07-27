
using DelimitedFiles
using Distributions
using LaTeXStrings
using LinearAlgebra

function nodal_value(var,scalar,Nx,Ny,Nz,n)

    nx = Int(n/Nx)
    ny = Int(n/Ny)
    nz = Int(n/Nz)

    count = 0

    for kk = 1:Nz
        k1 = 1 + nz*(kk-1)
        k2 = nz + nz*(kk-1)
        for jj = 1:Ny
            j1 = 1 + ny*(jj-1)
            j2 = ny + ny*(jj-1)
            for ii = 1:Nx
                i1 = 1 + nx*(ii-1)
                i2 = nx + nx*(ii-1)
                for k = k1:k2
                    for j = j1:j2
                        for i = i1:i2
                            count = count + 1
                            scalar[i,j,k] = var[count]
                        end
                    end
                end
            end
        end
    end

    return scalar, count
end


# Read data
N = 128
Nx = 8
Ny = 4
Nz = 4

var1 = []
var2 = []
I = 0
for i = 1:N
    global I, var1, var2
    I = i - 1
    VAR = readdlm("Case_H_P_128/nodal_values/nodal_values-10.00-$I")
    append!(var1,VAR[:,6])
    append!(var2,VAR[:,9])
end

n = 128
scalar1 = zeros(n,n,n)
scalar2 = zeros(n,n,n)

# Read the 128^3 supersaturation (Se) data
(scalar1,count) = nodal_value(var1,scalar1,Nx,Ny,Nz,n)

# Read the 128^3 cloud water micing ratio (ql) data
(scalar2,count) = nodal_value(var2,scalar2,Nx,Ny,Nz,n)

rho_l = 1000
h = 0.512/128
rho_a = 1.0
C = (4*pi*rho_l)/(3*rho_a*(h^3))

scalar2 .= C .*(scalar2.^3)



# Read data
N = 128
Nx = 8
Ny = 4
Nz = 4

var3 = []
var4 = []
I = 0
for i = 1:N
    global I, var1, var2
    I = i - 1
    VAR = readdlm("Case_H_P_192/nodal_values/nodal_values-10.00-$I")
    append!(var3,VAR[:,6])
    append!(var4,VAR[:,9])
end

n = 192
scalar3 = zeros(n,n,n)
scalar4 = zeros(n,n,n)

# Read the 192^3 supersaturation (Se) data
(scalar3,count) = nodal_value(var3,scalar3,Nx,Ny,Nz,n)

# Read the 192^3 cloud water micing ratio (ql) data
(scalar4,count) = nodal_value(var4,scalar4,Nx,Ny,Nz,n)

rho_l = 1000
h = 0.512/192
rho_a = 1.0
C = (4*pi*rho_l)/(3*rho_a*(h^3))

scalar4 .= C .*(scalar4.^3)
