#include "solve.h"
#include <cuda_runtime.h>

#define MAX_LENGTH 6

// FNV-1a hash function that takes a byte array and its length as input
// Returns a 32-bit unsigned integer hash value
__device__ unsigned int fnv1a_hash_bytes(const unsigned char* data, int length) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    for (int i = 0; i < length; i++) {
        hash = (hash ^ data[i]) * FNV_PRIME;
    }
    return hash;
}

__global__ void crack_password(unsigned int target_hash, 
    int password_length, int R, 
    char* output_password, unsigned long long num_passwords){
        unsigned long long index = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;

        // Bounds check due to ceil operation during kernel launch
        if(index >= num_passwords){
            return;
        }

        // Generating the password string
        // Thinking of index as a number with base 10 which needs to be converted into a system of base 26

        char curr_password[MAX_LENGTH+1];

        unsigned long long temp_index = index;

        for(int i = password_length-1; i >= 0; --i){
            int remainder = temp_index % 26;
            curr_password[i] = 'a' + remainder; // Since 'a' corresponds to 0 in the new system;
            temp_index = temp_index / 26;
        }

        // Adding null terminator at the end
        curr_password[password_length] = '\0';

        // Performing hashing on this password
        unsigned int curr_hash;

        curr_hash = fnv1a_hash_bytes((const unsigned char*)curr_password, password_length);

        // Now this needs to be hashed R-1 times
        // However curr_hash is unsigned int which needs to be converted to unsigned char *

        // Rounds 2 - R-1
        if(R > 1){
            unsigned char hash_bytes[4];

            for(int round = 2; round <= R; ++round){
                hash_bytes[3] = (curr_hash >> 24) & 0xFF;
                hash_bytes[2] = (curr_hash >> 16) & 0xFF;
                hash_bytes[1] = (curr_hash >> 8) & 0xFF;
                hash_bytes[0] = (curr_hash) & 0xFF;

                curr_hash = fnv1a_hash_bytes(hash_bytes, 4);
            }
        }

        if(curr_hash == target_hash){
            // write to output
            for(int i = 0; i < password_length; ++i){
                output_password[i] = curr_password[i];
            }

            output_password[password_length] = '\0';
        }

}

// output_password is a device pointer
void solve(unsigned int target_hash, int password_length, int R, char* output_password) {
    //int total_possible_combinations = pow(26, password_length); this works for floating point types. But for a large value we need lonbg
    unsigned long long num_passwords = 1;
    
    for(int i=0; i < password_length; ++i){
        num_passwords *= 26;
    }

    dim3 threadsPerBlock(256);
    dim3 gridDim((num_passwords + threadsPerBlock.x - 1)/ threadsPerBlock.x);

    crack_password<<<gridDim, threadsPerBlock>>>(
        target_hash, 
        password_length, 
        R, 
        output_password,
        num_passwords);

    cudaDeviceSynchronize();

    
}