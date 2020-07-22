//
//  main.swift
//  Bronze
//
//  Created by Bradley on 7/19/20.
//  Copyright © 2020 Bradley. All rights reserved.
//

import Foundation
import Metal
import simd

var bronze = Bronze()



var simds = [simd_float4x4]()
var count = 10000000
for j in 0..<count{
    let i = Float(j)
    simds.append(simd_float4x4(columns: (simd_float4(i,1*i,2*i,3*i), simd_float4(4,5,6,7), simd_float4(8,9,10,11*i), simd_float4(12*i,13,14,15))))
}

var multiplier = simd_float4x4(columns: (simd_float4(1,1,1,1), simd_float4(1,1,1,1), simd_float4(1,1,2,2), simd_float4(2,3,4,5)))


let multiplier_mat = bronze.newGPUMatFromSIMD(input: &multiplier)


let gpumat = bronze.newGPUMatArrayFromSIMD(input: simds)
let simds_returned = gpumat.toSIMD()

var start = DispatchTime.now()
bronze.multMatrixMultiInPlaceSquare(A: gpumat, B: multiplier_mat, right: false)
let out_simds = gpumat.toSIMD()
var end = DispatchTime.now()
print("GPU: ",Double(end.uptimeNanoseconds-start.uptimeNanoseconds)/1_000_000_000)


start = DispatchTime.now()
var out_simds_cpu = Array<simd_float4x4>()
for i in 0..<simds.count{
    out_simds_cpu.append(simd_mul(simds[i], multiplier))
}
end = DispatchTime.now()
print("CPU: ",Double(end.uptimeNanoseconds-start.uptimeNanoseconds)/1_000_000_000)
