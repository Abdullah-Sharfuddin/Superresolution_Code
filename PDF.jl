using DelimitedFiles
using Distributions
using Plots
using LaTeXStrings
using LinearAlgebra
using FFTW
using HDF5
using Statistics

# Function to compute PDF
function PDF_array(array,num,num_bins)

	var_min = 10^10
	var_max = -10^10

	array_size = num

	for i = 1:array_size
		if array[i] < var_min
			var_min = array[i]
		end
		if array[i] > var_max
			var_max = array[i]
		end
	end

	bin_size = (var_max - var_min)/(num_bins-1)
	
	prob_dens = zeros(num_bins,1)

	for i = 1:array_size
		for j = 1:num_bins
			if (array[i] >= var_min+(j-0.5)*bin_size) && (array[i] < var_min+(j+0.5)*bin_size)
				prob_dens[j] = prob_dens[j] + 1.0
				break
			end
		end
	end

	total_num = 0
	# Normalize Probability density function
	for j = 1:num_bins
		total_num = total_num + prob_dens[j]
	end

	for j = 1:num_bins
		prob_dens[j] = prob_dens[j]/(bin_size*total_num)
	end

	bin_mid = zeros(num_bins,1)

	for i = 1:num_bins
        bin_mid[i] = var_min+(0.5+i)*bin_size
    end

	return prob_dens, bin_mid
end


function nodal_value(var,scalar,Nx,Ny,Nz)

    n = 192
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
fid = h5open("Run_1/SeQl_192_flowfno.h5", "r")

Se_128_input    = read(fid["Se_128_input"])     # original 128³ Se (signed)
ql_128_input    = read(fid["ql_128_input"])     # original 128³ q_l (≥0)

Se_192_sample   = read(fid["Se_192_sample"])    # FlowFNO stochastic sample, Se
ql_192_sample   = read(fid["ql_192_sample"])    # FlowFNO stochastic sample, q_l

N = 128
Nx = 8
Ny = 4
Nz = 4

var = []
I = 0
for i = 1:N
    global I, var
    I = i - 1
    VAR = readdlm("Case_H_P_192/nodal_values/nodal_values-10.00-$I")
    append!(var,VAR[:,9])
end

n1 = 128
n2 = 192
scalar1 = ql_128_input 
scalar2 = ql_192_sample
scalar3 = zeros(n2,n2,n2)   ## For 192^3 DNS
(scalar3,count) = nodal_value(var,scalar3,Nx,Ny,Nz)

# For liquid water mixing ratio
rho_l = 1000
h = 0.512/192
rho_a = 1.0
C = (4*pi*rho_l)/(3*rho_a*(h^3))
for k = 1:n2
    for j = 1:n2
        for i = 1:n2
            scalar3[i,j,k] = C*(scalar3[i,j,k]^3)
        end
    end
end

# Compute mean
mean_scalar1 = sum(scalar1[:]) / (n1*n1*n1)
mean_scalar2 = sum(scalar2[:]) / (n2*n2*n2)
mean_scalar3 = sum(scalar3[:]) / (n2*n2*n2)

# Compute PDF
array1 = scalar1[:]
num_bins = 400
count1 = size(array1,1)
(prob_dens1, bin_mid1) = PDF_array(array1,count1,num_bins)

x1 = [mean_scalar1, mean_scalar1]; y1 = [0,1]

mean_err = (mean_scalar3 - mean_scalar2)/mean_scalar3
std2 = std(scalar2)
std3 = std(scalar3)
std_err = (std3 - std2)/std3


#=
plot(bin_mid1[:],log.(10,prob_dens1[:]),color = :red4,linestyle =:solid,linewidth=2,
labels=false,grid=false,thickness_scaling=1.5,tickfont=font(9,"Helvetica Bold"))
plot!(x1,y1, color = :red4,linestyle =:solid,linewidth=2,
grid=false,thickness_scaling=1.5,legend=false)


# Compute PDF
array2 = scalar2[:]
num_bins = 400
count2 = size(array2,1)
(prob_dens2, bin_mid2) = PDF_array(array2,count2,num_bins)

x2 = [mean_scalar2, mean_scalar2]; y2 = [0,1]

plot!(bin_mid2[:],log.(10,prob_dens2[:]),color = :blue4,linestyle =:dash,linewidth=2,
labels=false,grid=false,thickness_scaling=1.5,tickfont=font(9,"Helvetica Bold"))
plot!(x2,y2, color = :blue4,linestyle =:dash,linewidth=2,
grid=false,thickness_scaling=1.5,legend=false)


# Compute PDF
array3 = scalar3[:]
num_bins = 400
count3 = size(array3,1)
(prob_dens3, bin_mid3) = PDF_array(array3,count3,num_bins)

x3 = [mean_scalar3, mean_scalar3]; y3 = [0,1]

plot!(bin_mid3[:],log.(10,prob_dens3[:]),color = :green4,linestyle =:dot,linewidth=2.5,
labels=false,grid=false,thickness_scaling=1.5,tickfont=font(9,"Helvetica Bold"))
plot!(x3,y3, color = :green4,linestyle =:dot,linewidth=2.5,
grid=false,thickness_scaling=1.5,legend=false)

=#