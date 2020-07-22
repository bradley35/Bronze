//
//  matAdd.metal
//  Bronze
//
//  Created by Bradley on 7/19/20.
//  Copyright © 2020 Bradley. All rights reserved.
//

#include <metal_stdlib>

#include "random.h"

#define DECIMALS 6
#define POS_DIGITS 6
using namespace metal;

struct matAddVars {
    const device float* A [[buffer(0)]];
    const device float* B [[buffer(1)]];
    device float* C [[buffer(2)]];
};
constant int seed [[function_constant((0))]];

kernel void fillRandom(device float* out [[buffer(0)]], uint pos [[thread_position_in_grid]]){
    if(is_function_constant_defined(seed)){
        out[pos] = rand(pos, seed, seed<<1);
    }else{
        out[pos] = rand(pos, pos<<1, pos<<2);
    }
    
}
kernel void matAddSimple(matAddVars inputs, uint pos [[thread_position_in_grid]], uint index [[thread_index_in_threadgroup]]){ // One at a time
    float result = inputs.A[pos]+inputs.B[pos];
    inputs.C[pos] = result;
}

kernel void generateString(const device int* width [[buffer(0)]], device const float* input [[buffer(1)]], device char* outputs [[buffer(2)]], uint pos [[thread_position_in_grid]]){
    
    float current = input[pos];
    int power = log10(input[pos]);
    for (int i=POS_DIGITS; i>=-DECIMALS; i--){
        if(i>=0){
            if(i>power && i>0){
                outputs[(DECIMALS+POS_DIGITS+2)*pos+(POS_DIGITS-i)] = (char)7;
            }else{
                float divider = pow(10.0, i);
                int divided = current/divider;
                outputs[(DECIMALS+POS_DIGITS+2)*pos+(POS_DIGITS-i)] = (char)(48 + divided);
                current = current-divider*divided;
            }
        }
        else if (i == -1){
            outputs[(DECIMALS+POS_DIGITS+2)*pos+(POS_DIGITS-i)] = '.';
        }else{
            float divider = pow(10.0, i+1);
            int divided = current/divider;
            outputs[(DECIMALS+POS_DIGITS+2)*pos+(POS_DIGITS-i)] = (char)(48 + divided);
            current = current-divider*divided;
        }
    }
    if((pos+1)%width[0] == 0){
        outputs[(DECIMALS+POS_DIGITS+2)*(pos+1)-1] = '\n';
    }else{
        outputs[(DECIMALS+POS_DIGITS+2)*(pos+1)-1] = ',';
    }
    
}

kernel void matMultScalar(device const float* mat [[buffer(0)]], device const float* scalar [[buffer(1)]], device float* out [[buffer(2)]], uint pos [[thread_position_in_grid]]){
    out[pos] = mat[pos]*scalar[0];
}

void matMultSimpleFunc(const device float* A, const device float* B, const device int* Awidth, const device int* Bwidth, device float* out, uint2 pos){
    uint x = pos[0];//Position in output
    uint y = pos[1];//Position in output
    
    uint index_out = (*Bwidth) * y + x;//Output matrix has the same width as B
    
    float sum = 0;
    
    for (int Aindex = 0; Aindex < *Awidth; Aindex++){
        sum += A[y*(*Awidth) + Aindex] * B[x + *Bwidth*Aindex];
    }
    
    out[index_out] = sum;
    
    
}
kernel void matMultSimple(const device float* A [[buffer(0)]], const device float* B [[buffer(1)]], const device int* Awidth [[buffer(2)]], const device int* Bwidth [[buffer(3)]], device float* out [[buffer(4)]], uint2 pos [[thread_position_in_grid]]){
    matMultSimpleFunc(A,B,Awidth,Bwidth,out,pos);
}
struct matSize{
    const device int* Awidth [[buffer(2)]];
    const device int* Aheight [[buffer(3)]];
    const device int* Bwidth [[buffer(4)]];
    const device int* Bheight [[buffer(5)]];
};
kernel void matMultMultiRight(const device float* A [[buffer(0)]], const device float* B [[buffer(1)]], matSize size, device float* out [[buffer(6)]], uint pos_in_grid [[thread_position_in_grid]]){//Multiply XxYxZ matrix by one AxX matrix repeated; Each threadgroup is an instance of the initial matrix
    //Computes A x B
    
    int count = pos_in_grid/((*size.Bwidth)*(*size.Aheight));
    
    device const float* Ashifted = &A[count*(*size.Awidth)*(*size.Aheight)];
    device float* Outshifted = &out[count*(*size.Bwidth)*(*size.Aheight)];
    
    int position = pos_in_grid % ((*size.Bwidth)*(*size.Aheight));
    int x = position % (*size.Bwidth);
    int y = position / (*size.Bwidth);
    matMultSimpleFunc(Ashifted, B, size.Awidth, size.Bwidth, Outshifted, uint2(x, y));
}
kernel void matMultMultiLeft(const device float* A [[buffer(0)]], const device float* B [[buffer(1)]], matSize size, device float* out [[buffer(6)]], uint pos_in_grid [[thread_position_in_grid]]){//Multiply XxYxZ matrix by one AxX matrix repeated; Each threadgroup is an instance of the initial matrix
    //Computes B x A
    
    int count = pos_in_grid/((*size.Awidth)*(*size.Bheight));
    
    device const float* Ashifted = &A[count*(*size.Awidth)*(*size.Aheight)];
    device float* Outshifted = &out[count*(*size.Awidth)*(*size.Bheight)];
    
    int position = pos_in_grid % ((*size.Awidth)*(*size.Bheight));
    int x = position % (*size.Awidth);
    int y = position / (*size.Awidth);
    matMultSimpleFunc(B, Ashifted, size.Bwidth, size.Awidth, Outshifted, uint2(x, y));
}
//kernel void matMultMultiLeft(const device float* A [[buffer(0)]], const device float* B [[buffer(1)]], uint2 pos_in_group [[thread_position_in_threadgroup]], uint group_in_grid [[threadgroup_position_in_grid]]){//Multiply AxX matrix repeated by XxYxZ matrix; Each threadgroup is an instance of the initial matrix
//    //Computes B x A
//
//}
