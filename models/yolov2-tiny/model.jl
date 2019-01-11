using Statistics 
Mul(a,b,c) = b .* reshape(c, (1,1,size(c)[a],1)) 
Add(axis, A ,B) = A .+ reshape(B, (1,1,size(B)[1],1)) 
flipkernel(x) = x[end:-1:1, end:-1:1, :, :] 

begin
    c_1 = Conv((1,1),1024=>125, relu, stride=(1, 1), pad=(0, 0), dilation=(1, 1))
    c_2 = BatchNorm(1024,identity)
    c_3 = Conv((3,3),1024=>1024, relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_4 = BatchNorm(1024,identity)
    c_5 = Conv((3,3),512=>1024, relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_6 = BatchNorm(512,identity)
    c_7 = Conv((3,3),256=>512, relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_8 = BatchNorm(256,identity)
    c_9 = Conv((3,3),128=>256, relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_10 = BatchNorm(128,identity)
    c_11 = Conv((3,3),64=>128, relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_12 = BatchNorm(64,identity)
    c_13 = Conv((3,3),32=>64, relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_14 = BatchNorm(32,identity)
    c_15 = Conv((3,3),16=>32, relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
    c_16 = BatchNorm(16,identity)
    c_17 = Conv((3,3),3=>16, relu, stride=(1, 1), pad=(1, 1), dilation=(1, 1))
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

    