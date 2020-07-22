//
//  Bronze.swift
//  Bronze
//
//  Created by Bradley on 7/20/20.
//  Copyright © 2020 Bradley. All rights reserved.
//

import Foundation
import Metal
import simd

class Bronze{
    var device:MTLDevice
    var library:MTLLibrary
    
    var storageMode:MTLResourceOptions
    
    fileprivate var id:String
    
    lazy var matAddSimple:MTLComputePipelineState = try! device.makeComputePipelineState(function: library.makeFunction(name: "matAddSimple")!)
    lazy var matMultScalar:MTLComputePipelineState = try! device.makeComputePipelineState(function: library.makeFunction(name: "matMultScalar")!)
    lazy var matMultSimple:MTLComputePipelineState = try! device.makeComputePipelineState(function: library.makeFunction(name: "matMultSimple")!)
    lazy var matMultMultiRight:MTLComputePipelineState = try! device.makeComputePipelineState(function: library.makeFunction(name: "matMultMultiRight")!)
    lazy var matMultMultiLeft:MTLComputePipelineState = try! device.makeComputePipelineState(function: library.makeFunction(name: "matMultMultiLeft")!)
    
    var randomConstants:MTLFunctionConstantValues {
        var useed:UInt32 = arc4random()
        var seed:Int32 = 0
        if(useed>Int32.max){
            seed = -Int32(useed-UInt32(Int32.max))
        }else{
            seed = Int32(useed)
        }
        let values = MTLFunctionConstantValues()
        values.setConstantValue(&seed, type: .int, index: 0)
        return values
    }
    var fillRandom:MTLComputePipelineState{
        try! device.makeComputePipelineState(function: library.makeFunction(name: "fillRandom", constantValues: randomConstants))
    }
    
    lazy var generateString:MTLComputePipelineState = try! device.makeComputePipelineState(function: library.makeFunction(name: "generateString")!)
    
    var max_tgs:MTLSize
    
    var queue:MTLCommandQueue
    
    init(){
        self.device = MTLCreateSystemDefaultDevice()!
         #if os(macOS)
        for device in MTLCopyAllDevices(){
            if(!device.isLowPower){
                self.device = device
                break
            }
        }
        #endif
        self.library = device.makeDefaultLibrary()!
        self.queue = device.makeCommandQueue()!
        
        self.max_tgs = device.maxThreadsPerThreadgroup
        
        self.id = UUID().uuidString
        #if os(macOS)
        storageMode = .storageModeManaged
        #else
        storageMode = .storageModeShared
        #endif
    }
    
    func generateRandomMatrix(width:Int32, height:Int32)->GPUMat{
        var input = newGPUMat(width: width, height: height)
        let cmdBuffer = queue.makeCommandBuffer()!
        
        let commandEncoder = cmdBuffer.makeComputeCommandEncoder()!
        
        commandEncoder.setComputePipelineState(fillRandom)
        commandEncoder.setBuffer(input.buffer, offset: 0, index: 0)
        
        let gridSize = MTLSizeMake(Int(width*height), 1, 1)
        
        let threadGroupSize = MTLSizeMake(min(Int(width*height), max_tgs.width), 1, 1)
        
        commandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize) // Generate random matrix in GPU memory
        commandEncoder.endEncoding()
        
        #if os(macOS)
            let blitEncoder = cmdBuffer.makeBlitCommandEncoder()! // Get back randomized matrix in CPU memory
            blitEncoder.synchronize(resource: input.buffer)
            blitEncoder.endEncoding()
        #endif
        
        cmdBuffer.commit()
        
        
        cmdBuffer.waitUntilCompleted()
        
        return input
    }
    func stringFromMatrix(mat:GPUMat) ->String{
        let output = device.makeBuffer(length: Int(Int32(MemoryLayout<CChar>.size)*mat.width*mat.height*14), options: storageMode)!
        var width = mat.width
        let width_buffer = device.makeBuffer(bytes: &width, length: MemoryLayout<Int32>.size, options: storageMode)!
        
        let cmdBuffer = queue.makeCommandBuffer()!
        let commandEncoder = cmdBuffer.makeComputeCommandEncoder()!
        
        commandEncoder.setComputePipelineState(generateString)
        commandEncoder.setBuffer(width_buffer, offset: 0, index: 0)
        commandEncoder.setBuffer(mat.buffer, offset: 0, index: 1)
        commandEncoder.setBuffer(output, offset: 0, index: 2)
        let gridSize = MTLSizeMake(Int(mat.width*mat.height), 1, 1)
        let threadGroupSize = MTLSizeMake(min(Int(mat.width*mat.height), max_tgs.width), 1, 1)
        commandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        commandEncoder.endEncoding()
        
        #if os(macOS)
            let blitEncoder = cmdBuffer.makeBlitCommandEncoder()! // Get back randomized matrix in CPU memory
            blitEncoder.synchronize(resource: output)
            blitEncoder.endEncoding()
        #endif
        
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
        
        return String(cString: output.contents().assumingMemoryBound(to: CChar.self))
    }
    func printMatrix(mat:GPUMat){
        print(stringFromMatrix(mat: mat))
        
    }
    func addMatrices(A:GPUMat, B:GPUMat) -> GPUMat{
        if(A.width != B.width || A.height != B.height){
            fatalError("Matrices must be the same size")
        }
        
        let output = newGPUMat(width: A.width, height: A.height)
        
        let cmdBuffer = queue.makeCommandBuffer()!
        let commandEncoder = cmdBuffer.makeComputeCommandEncoder()!
        
        commandEncoder.setComputePipelineState(matAddSimple)
        commandEncoder.setBuffer(A.buffer, offset: 0, index: 0)
        commandEncoder.setBuffer(B.buffer, offset: 0, index: 1)
        commandEncoder.setBuffer(output.buffer, offset: 0, index: 2)
        
        let gridSize = MTLSizeMake(Int(output.width*output.height), 1, 1)
        let threadGroupSize = MTLSizeMake(min(Int(output.width*output.height), max_tgs.width), 1, 1)
        commandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        commandEncoder.endEncoding()
        
        #if os(macOS)
            let blitEncoder = cmdBuffer.makeBlitCommandEncoder()! // Get back randomized matrix in CPU memory
            blitEncoder.synchronize(resource: output.buffer)
            blitEncoder.endEncoding()
        #endif
        
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
        
        return output
    }
    
    func multMatrixScalar(A:GPUMat, scalar:Float32) -> GPUMat{
        let output = newGPUMat(width: A.width, height: A.height)
        let cmdBuffer = queue.makeCommandBuffer()!
        let commandEncoder = cmdBuffer.makeComputeCommandEncoder()!
        commandEncoder.setComputePipelineState(matMultScalar)
        
        var scalarRef = scalar
        let scalarBuffer = device.makeBuffer(bytes: &scalarRef, length: MemoryLayout<Float32>.size, options: storageMode)
        
        commandEncoder.setBuffer(A.buffer, offset: 0, index: 0)
        commandEncoder.setBuffer(scalarBuffer, offset: 0, index: 1)
        commandEncoder.setBuffer(output.buffer, offset: 0, index: 2)
        
        let gridSize = MTLSizeMake(Int(output.width*output.height), 1, 1)
        let threadGroupSize = MTLSizeMake(min(Int(output.width*output.height), max_tgs.width), 1, 1)
        commandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        commandEncoder.endEncoding()
        #if os(macOS)
            let blitEncoder = cmdBuffer.makeBlitCommandEncoder()! // Get back randomized matrix in CPU memory
            blitEncoder.synchronize(resource: output.buffer)
            blitEncoder.endEncoding()
        #endif

        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
        
        return output
    }
    
    func multMatrix(A:GPUMat, B:GPUMat) -> GPUMat{
        let output = newGPUMat(width: B.width, height: A.height)
        let cmdBuffer = queue.makeCommandBuffer()!
        let commandEncoder = cmdBuffer.makeComputeCommandEncoder()!
        commandEncoder.setComputePipelineState(matMultSimple)
        
        var A_width = A.width
        var B_width = B.width
        
        let A_width_buffer = device.makeBuffer(bytes: &A_width, length: MemoryLayout<Int32>.size, options: storageMode)
        let B_width_buffer = device.makeBuffer(bytes: &B_width, length: MemoryLayout<Int32>.size, options: storageMode)
        
        commandEncoder.setBuffer(A.buffer, offset: 0, index: 0)
        commandEncoder.setBuffer(B.buffer, offset: 0, index: 1)
        commandEncoder.setBuffer(A_width_buffer, offset: 0, index: 2)
        commandEncoder.setBuffer(B_width_buffer, offset: 0, index: 3)
        
        commandEncoder.setBuffer(output.buffer, offset: 0, index: 4)
        
        let gridSize = MTLSizeMake(Int(output.width), Int(output.height), 1)
        
        let threadGroupWidth = min(Int(output.width), max_tgs.width)
        let threadGroupSize = MTLSizeMake(threadGroupWidth, max(max_tgs.height/threadGroupWidth,1), 1)
        commandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        commandEncoder.endEncoding()
        #if os(macOS)
            let blitEncoder = cmdBuffer.makeBlitCommandEncoder()! // Get back multiplied matrix in CPU memory
            blitEncoder.synchronize(resource: output.buffer)
            blitEncoder.endEncoding()
        #endif
        
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
        
        return output
    }
    func multMatrixMulti(A:GPUMatArray, B:GPUMat) -> GPUMatArray{//Max elements per output matrix is 1,024
        let output = newBlankGPUMatArray(width: B.width, height: A.height, count: A.count)
        print("Done Making Output")
        let cmdBuffer = queue.makeCommandBuffer()!
        let commandEncoder = cmdBuffer.makeComputeCommandEncoder()!
        commandEncoder.setComputePipelineState(matMultMultiRight)
        
        var Awidth = A.width; let Awidth_buffer = device.makeBuffer(bytes: &Awidth, length: MemoryLayout<Int32>.size, options: storageMode)!
        var Aheight = A.height; let Aheight_buffer = device.makeBuffer(bytes: &Aheight, length: MemoryLayout<Int32>.size, options: storageMode)!
        
        var Bwidth = B.width; let Bwidth_buffer = device.makeBuffer(bytes: &Bwidth, length: MemoryLayout<Int32>.size, options: storageMode)!
        var Bheight = B.height; let Bheight_buffer = device.makeBuffer(bytes: &Bheight, length: MemoryLayout<Int32>.size, options: storageMode)!
        
        //A,B,Awidth,Aheight,Bwidth,Bheight, out
        
        commandEncoder.setBuffers([A.buffer, B.buffer,Awidth_buffer, Aheight_buffer, Bwidth_buffer, Bheight_buffer, output.buffer], offsets: [0,0,0,0,0,0,0], range: 0..<7)
        
        let gridSize = MTLSizeMake(Int(output.width) * Int(output.height)*Int(A.count),1,1)
        let threadGroupSize = MTLSizeMake(min(Int(output.width) * Int(output.height)*Int(A.count), max_tgs.width), 1, 1)
        commandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        commandEncoder.endEncoding()
        
        #if os(macOS)
            let blitEncoder = cmdBuffer.makeBlitCommandEncoder()! // Get back multiplied matrix in CPU memory
            blitEncoder.synchronize(resource: output.buffer)
            blitEncoder.endEncoding()
        #endif
        
        cmdBuffer.commit()
        print("Commited")
        //let start = DispatchTime.now()
        cmdBuffer.waitUntilCompleted()
        //let end = DispatchTime.now()
        //print("Time: ",Double(end.uptimeNanoseconds-start.uptimeNanoseconds)/1_000_000_000)
        
        return output
    }
    func multMatrixMultiInPlaceSquare(A:GPUMatArray, B:GPUMat, right:Bool = true){
        if !((A.width == A.height) && (B.width == B.height) && (A.width == B.width)){
            fatalError("Not Square")
        }
        let cmdBuffer = queue.makeCommandBuffer()!
        let commandEncoder = cmdBuffer.makeComputeCommandEncoder()!
        if(right){
            commandEncoder.setComputePipelineState(matMultMultiRight)
        }else{
            commandEncoder.setComputePipelineState(matMultMultiLeft)
        }
        
        
        var Awidth = A.width; let Awidth_buffer = device.makeBuffer(bytes: &Awidth, length: MemoryLayout<Int32>.size, options: storageMode)!
        var Aheight = A.height; let Aheight_buffer = device.makeBuffer(bytes: &Aheight, length: MemoryLayout<Int32>.size, options: storageMode)!
        
        var Bwidth = B.width; let Bwidth_buffer = device.makeBuffer(bytes: &Bwidth, length: MemoryLayout<Int32>.size, options: storageMode)!
        var Bheight = B.height; let Bheight_buffer = device.makeBuffer(bytes: &Bheight, length: MemoryLayout<Int32>.size, options: storageMode)!
        
        //A,B,Awidth,Aheight,Bwidth,Bheight, out
        
        commandEncoder.setBuffers([A.buffer, B.buffer,Awidth_buffer, Aheight_buffer, Bwidth_buffer, Bheight_buffer, A.buffer], offsets: [0,0,0,0,0,0,0], range: 0..<7)
        
        let gridSize = MTLSizeMake(Int(A.width) * Int(A.height)*Int(A.count),1,1)
        let threadGroupSize = MTLSizeMake(min(Int(A.width) * Int(A.height)*Int(A.count), max_tgs.width), 1, 1)
        commandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        commandEncoder.endEncoding()
        
        #if os(macOS)
            let blitEncoder = cmdBuffer.makeBlitCommandEncoder()! // Get back multiplied matrix in CPU memory
            blitEncoder.synchronize(resource: A.buffer)
            blitEncoder.endEncoding()
        #endif
        
        cmdBuffer.commit()
        //print("Commited")
        //let start = DispatchTime.now()
        cmdBuffer.waitUntilCompleted()
        //let end = DispatchTime.now()
        //print("Time: ",Double(end.uptimeNanoseconds-start.uptimeNanoseconds)/1_000_000_000)
        
    }
    
    
    func newGPUMat(width:Int32, height:Int32) -> GPUMat{
        return GPUMat.zeros(width: width, height: height, bronzeParent: self)
    }
    
    func newGPUMatArrayFromGPUMats(_ mats:[GPUMat]) -> GPUMatArray{
        return GPUMatArray.fromGPUMats(mats: mats)
    }
    func newBlankGPUMatArray(width:Int32, height:Int32, count:Int32) -> GPUMatArray{
        return GPUMatArray.zeros(width: width, height: height, count: count, bronzeParent: self)
    }
    func newGPUMatArrayFromSIMD(input:[simd_float4x4]) -> GPUMatArray{
        return GPUMatArray.fromSIMD(input: input, bronzeParent: self)
    }
    func newGPUMatFromSIMD(input: inout simd_float4x4) -> GPUMat{
        return GPUMat.fromSIMD(input: &input, bronzeParent: self)
    }
}

extension Bronze{
    struct GPUMat{
        var width:Int32
        var height:Int32
        var buffer:MTLBuffer
        fileprivate var bronzeParent:Bronze
        
        fileprivate static func zeros(width:Int32, height:Int32, bronzeParent:Bronze) -> GPUMat{
            //print(bronzeParent.device.maxBufferLength)
            return GPUMat(width: width, height: height, buffer: bronzeParent.device.makeBuffer(length: Int(height)*Int(width)*MemoryLayout<Float>.size, options: bronzeParent.storageMode)!, bronzeParent: bronzeParent)
        }
        fileprivate static func fromSIMD(input: inout simd_float4x4, bronzeParent:Bronze) -> GPUMat{
            
            let buffer = bronzeParent.device.makeBuffer(bytes: &input, length: 16*MemoryLayout<Float32>.size, options: bronzeParent.storageMode)!
            
            return GPUMat(width: 4, height: 4, buffer: buffer, bronzeParent: bronzeParent)
        }
        
        static func + (left:GPUMat, right:GPUMat) -> GPUMat{
            if(left.bronzeParent.id != right.bronzeParent.id){
                fatalError("Adding two matrices from different bronze instances")
            }
            
            return left.bronzeParent.addMatrices(A: left, B: right)
        }
        
        static func * (left:GPUMat, right:Float32) -> GPUMat{
            return left.bronzeParent.multMatrixScalar(A: left, scalar: right)
        }
        static func * (left:GPUMat, right:GPUMat) -> GPUMat{
            if(left.bronzeParent.id != right.bronzeParent.id){
                fatalError("Two different bronze instances")
            }
            if(left.width != right.height){
                fatalError("Cannot multiply incompatible matrices")
            }
            return left.bronzeParent.multMatrix(A: left, B: right)
        }
        subscript (y: Int, x:Int) -> Float32{
            get{
                return buffer.contents().assumingMemoryBound(to: Float32.self).advanced(by: Int(width)*y + x).pointee
            }
            set (item){
                var position = Int(width)*y + x
                buffer.contents().assumingMemoryBound(to: Float32.self).advanced(by: position)[0] = item
                buffer.didModifyRange(position*MemoryLayout<Float32>.size..<(position+1)*MemoryLayout<Float32>.size)
            }
        }
        
    }
    struct GPUMatArray{
        var width:Int32
        var height:Int32
        var buffer:MTLBuffer
        fileprivate var bronzeParent:Bronze
        var count:Int32
        
        fileprivate static func fromGPUMats(mats:[GPUMat]) -> GPUMatArray{//All matrices must be the same shape and from the same Bronze instance
            let width = mats[0].width
            let height = mats[0].height
            
            let buffer = mats[0].bronzeParent.device.makeBuffer(length: Int(width*height*Int32(mats.count)*Int32(MemoryLayout<Float32>.size)), options: mats[0].bronzeParent.storageMode)!
            
            var current_pointer = buffer.contents()
            
            for mat in mats{
                current_pointer.copyMemory(from: mat.buffer.contents(), byteCount: Int(width*height*Int32(MemoryLayout<Float32>.size)))
                current_pointer = current_pointer.advanced(by: Int(width*height*Int32(MemoryLayout<Float32>.size)))
            }
            buffer.didModifyRange(0..<buffer.length)
            
            return GPUMatArray(width: width, height: height, buffer: buffer, bronzeParent: mats[0].bronzeParent, count: Int32(mats.count))
        }
        
        func copyGPUMat(index:Int)->GPUMat{
            return GPUMat(width: width, height: height, buffer: bronzeParent.device.makeBuffer(bytes: buffer.contents().advanced(by: Int(width*height*Int32(index)*Int32(MemoryLayout<Float32>.size))), length: Int(width*height*Int32(MemoryLayout<Float32>.size)), options: bronzeParent.storageMode)!, bronzeParent: bronzeParent)
        }
        fileprivate static func zeros(width:Int32, height:Int32, count:Int32, bronzeParent:Bronze) -> GPUMatArray{
            //print(bronzeParent.device.maxBufferLength)
            return GPUMatArray(width: width, height: height, buffer: bronzeParent.device.makeBuffer(length: Int(width*height*count*Int32(MemoryLayout<Float32>.size)), options: bronzeParent.storageMode)!, bronzeParent: bronzeParent, count: count)
        }
        
        fileprivate static func fromSIMD(input:[simd_float4x4], bronzeParent:Bronze) -> GPUMatArray{
            let buffer = input.withUnsafeBufferPointer { (pointer) -> MTLBuffer in
                bronzeParent.device.makeBuffer(bytes: pointer.baseAddress!, length: 16*MemoryLayout<Float32>.size*input.count, options: bronzeParent.storageMode)!
            }
            
            return GPUMatArray(width: 4, height: 4, buffer: buffer, bronzeParent: bronzeParent, count: Int32(input.count))
        }
        
        func toSIMD() -> [simd_float4x4]{
            return Array<simd_float4x4>(unsafeUninitializedCapacity: Int(count)) { (pointer, counter) in
                counter = Int(count)
                pointer.baseAddress!.assign(from: buffer.contents().assumingMemoryBound(to: simd_float4x4.self), count: Int(count))
            }
            //buffer.contents().assumingMemoryBound(to: Float32.self).
        }
    }
    
    
}
