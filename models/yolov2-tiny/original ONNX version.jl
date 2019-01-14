using Statistics 
Mul(a,b,c) = b .* reshape(c, (1,1,size(c)[a],1)) 
Add(axis, A ,B) = A .+ reshape(B, (1,1,size(B)[1],1)) 
flipkernel(x) = x[end:-1:1, end:-1:1, :, :] 
begin
    c_1 = Conv(flipkernel(weights["convolution8_W"]), Float32[0.0], relu, stride=(1, 1), pad=(0, 0), dilation=(1, 1))
    c_2 = BatchNorm(identity, weights["BatchNormalization_B7"], weights["BatchNormalization_scale7"], broadcast(Float32, weights["BatchNormalization_mean7"]), broadcast(Float32, broadcast(sqrt, broadcast(+, 1.0f-5, weights["BatchNormalization_variance7"]))), 1.0f-5, 0.0f0, false)
    c_3 = Conv(flipkernel(weights["convolution7_W"]), Float32[0.0], relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_4 = BatchNorm(identity, weights["BatchNormalization_B6"], weights["BatchNormalization_scale6"], broadcast(Float32, weights["BatchNormalization_mean6"]), broadcast(Float32, broadcast(sqrt, broadcast(+, 1.0f-5, weights["BatchNormalization_variance6"]))), 1.0f-5, 0.0f0, false)
    c_5 = Conv(flipkernel(weights["convolution6_W"]), Float32[0.0], relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_6 = BatchNorm(identity, weights["BatchNormalization_B5"], weights["BatchNormalization_scale5"], broadcast(Float32, weights["BatchNormalization_mean5"]), broadcast(Float32, broadcast(sqrt, broadcast(+, 1.0f-5, weights["BatchNormalization_variance5"]))), 1.0f-5, 0.0f0, false)
    c_7 = Conv(flipkernel(weights["convolution5_W"]), Float32[0.0], relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_8 = BatchNorm(identity, weights["BatchNormalization_B4"], weights["BatchNormalization_scale4"], broadcast(Float32, weights["BatchNormalization_mean4"]), broadcast(Float32, broadcast(sqrt, broadcast(+, 1.0f-5, weights["BatchNormalization_variance4"]))), 1.0f-5, 0.0f0, false)
    c_9 = Conv(flipkernel(weights["convolution4_W"]), Float32[0.0], relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_10 = BatchNorm(identity, weights["BatchNormalization_B3"], weights["BatchNormalization_scale3"], broadcast(Float32, weights["BatchNormalization_mean3"]), broadcast(Float32, broadcast(sqrt, broadcast(+, 1.0f-5, weights["BatchNormalization_variance3"]))), 1.0f-5, 0.0f0, false)
    c_11 = Conv(flipkernel(weights["convolution3_W"]), Float32[0.0], relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_12 = BatchNorm(identity, weights["BatchNormalization_B2"], weights["BatchNormalization_scale2"], broadcast(Float32, weights["BatchNormalization_mean2"]), broadcast(Float32, broadcast(sqrt, broadcast(+, 1.0f-5, weights["BatchNormalization_variance2"]))), 1.0f-5, 0.0f0, false)
    c_13 = Conv(flipkernel(weights["convolution2_W"]), Float32[0.0], relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_14 = BatchNorm(identity, weights["BatchNormalization_B1"], weights["BatchNormalization_scale1"], broadcast(Float32, weights["BatchNormalization_mean1"]), broadcast(Float32, broadcast(sqrt, broadcast(+, 1.0f-5, weights["BatchNormalization_variance1"]))), 1.0f-5, 0.0f0, false)
    c_15 = Conv(flipkernel(weights["convolution1_W"]), Float32[0.0], relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_16 = BatchNorm(identity, weights["BatchNormalization_B"], weights["BatchNormalization_scale"], broadcast(Float32, weights["BatchNormalization_mean"]), broadcast(Float32, broadcast(sqrt, broadcast(+, 1.0f-5, weights["BatchNormalization_variance"]))), 1.0f-5, 0.0f0, false)
    c_17 = Conv(flipkernel(weights["convolution_W"]), Float32[0.0], relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    (x_18,)->begin
            edge_19 = x_18 .* 0.003921569f0
            edge_20 = maxpool(broadcast(leakyrelu, c_16(c_17(getindex(reshape(edge_19, size(edge_19, 1), size(edge_19, 2), Int(size(edge_19, 3) / 1), 1, size(edge_19, 4)), Colon(), Colon(), Colon(), 1, Colon()))), 0.1f0), (2, 2), pad=(0, 0), stride=(2, 2))
            edge_21 = maxpool(broadcast(leakyrelu, c_14(c_15(getindex(reshape(edge_20, size(edge_20, 1), size(edge_20, 2), Int(size(edge_20, 3) / 1), 1, size(edge_20, 4)), Colon(), Colon(), Colon(), 1, Colon()))), 0.1f0), (2, 2), pad=(0, 0), stride=(2, 2))
            edge_22 = maxpool(broadcast(leakyrelu, c_12(c_13(getindex(reshape(edge_21, size(edge_21, 1), size(edge_21, 2), Int(size(edge_21, 3) / 1), 1, size(edge_21, 4)), Colon(), Colon(), Colon(), 1, Colon()))), 0.1f0), (2, 2), pad=(0, 0), stride=(2, 2))
            edge_23 = maxpool(broadcast(leakyrelu, c_10(c_11(getindex(reshape(edge_22, size(edge_22, 1), size(edge_22, 2), Int(size(edge_22, 3) / 1), 1, size(edge_22, 4)), Colon(), Colon(), Colon(), 1, Colon()))), 0.1f0), (2, 2), pad=(0, 0), stride=(2, 2))
            edge_24 = maxpool(broadcast(leakyrelu, c_8(c_9(getindex(reshape(edge_23, size(edge_23, 1), size(edge_23, 2), Int(size(edge_23, 3) / 1), 1, size(edge_23, 4)), Colon(), Colon(), Colon(), 1, Colon()))), 0.1f0), (2, 2), pad=(0, 0), stride=(2, 2))
            edge_25 = maxpool(broadcast(leakyrelu, c_6(c_7(getindex(reshape(edge_24, size(edge_24, 1), size(edge_24, 2), Int(size(edge_24, 3) / 1), 1, size(edge_24, 4)), Colon(), Colon(), Colon(), 1, Colon()))), 0.1f0), (2, 2), pad=(0, 0), stride=(1, 1))
            edge_26 = broadcast(leakyrelu, c_4(c_5(getindex(reshape(edge_25, size(edge_25, 1), size(edge_25, 2), Int(size(edge_25, 3) / 1), 1, size(edge_25, 4)), Colon(), Colon(), Colon(), 1, Colon()))), 0.1f0)
            edge_27 = broadcast(leakyrelu, c_2(c_3(getindex(reshape(edge_26, size(edge_26, 1), size(edge_26, 2), Int(size(edge_26, 3) / 1), 1, size(edge_26, 4)), Colon(), Colon(), Colon(), 1, Colon()))), 0.1f0)
            c_1(getindex(reshape(edge_27, size(edge_27, 1), size(edge_27, 2), Int(size(edge_27, 3) / 1), 1, size(edge_27, 4)), Colon(), Colon(), Colon(), 1, Colon()))
        end
end