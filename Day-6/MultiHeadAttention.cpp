#include <iostream>
#include <limits>
#include <cmath> 
#include <vector>

using namespace std;

// Matrix multiplication Calculation
void matMul(const vector<float> &A, const vector <float> &B, vector <float> &C, int M, int K, int N)
{
    for(int i = 0; i < M; i++){
        for(int j=0; j < N; j++){
            float sum = 0.0f;
            for(int k =0; k < K; k++){
                sum += A[i*K+k]*B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

void softmax(vector<float> &row){
    float max_val = -numeric_limits<float>::infinity();
    
    for(float x:row){
        max_val = max(max_val, x);
    }
    
    // Subtracting max value from each item for numerical stability
    float sum =0.0f;
    for(float &x:row){
        x = exp(x-max_val);
        sum += x;
    }
    
    for(float &x:row){
        x /= sum;
    }
}

vector<float> multiHeadAttentionCPU(
    const vector<float>& Q,
    const vector<float>& K, 
    const vector<float>& V,
    const vector<float>& Wq,
    const vector<float>& Wk,
    const vector<float>& Wv,
    int seq_len, int d_model, int num_heads
){
    int d_head = d_model / num_heads;
    
    // Q, K and V projections
    vector<float> Q_proj(seq_len*d_model, 0.0f);
    vector<float> K_proj(seq_len*d_model, 0.0f);
    vector<float> V_proj(seq_len*d_model, 0.0f);
    
    matMul(Q, Wq, Q_proj, seq_len, d_model, d_model);
    matMul(K, Wk, K_proj, seq_len, d_model, d_model);
    matMul(V, Wv, V_proj, seq_len, d_model, d_model);
    
    vector<float> output(seq_len*d_model,0.0f);
    
    //Computing attention for each head 
    for(int h = 0; h < num_heads; h++){
        int head_start = h*d_head;
        
        // This loop checks attention of current word with every other word in sequence
        for(int i=0; i < seq_len; i++){
            vector<float> scores(seq_len, 0.0f);
            for(int j =0; j < seq_len; j++){
                float dot = 0.0f;
                for(int k = head_start;k < head_start+d_head; k++){
                    dot += Q_proj[i*d_model + k] * K_proj[j*d_model + k];
                }
                scores[j] = dot / sqrt(static_cast<float>(d_head));
            }
            
            softmax(scores);
            
            for(int k=head_start; k < head_start+d_head; k++){
                float out_val =0.0f;
                for(int j =0; j < seq_len; j++){
                    out_val += scores[j] * V_proj[j*d_model+k];
                }
                output[i*d_model + k] = out_val;
            }
        }
    }
    return output;
}


int main() {
    int seq_len = 4;
    int d_model = 8;
    int num_heads = 2;

    std::vector<float> Q(seq_len * d_model, 0.01f);
    std::vector<float> K(seq_len * d_model, 0.02f);
    std::vector<float> V(seq_len * d_model, 0.03f);
    std::vector<float> Wq(d_model * d_model, 1.0f);
    std::vector<float> Wk(d_model * d_model, 1.0f);
    std::vector<float> Wv(d_model * d_model, 1.0f);

    std::vector<float> output = multiHeadAttentionCPU(Q, K, V, Wq, Wk, Wv, seq_len, d_model, num_heads);

    std::cout << "Multi-Head Attention Output (CPU):\n";
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_model; j++) {
            std::cout << output[i * d_model + j] << " ";
        }
        std::cout << "\n";
    }
    return 0;
}
