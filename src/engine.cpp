#include "loader.hpp"
#include "engine.hpp"

#include <iostream> 
#include <vector>
#include <bit>
#include <fstream>
#include <Eigen/Dense>
#include <format>
#include <climits>
#include <cassert>
#include <cmath>

using namespace std;
using namespace Eigen;

constexpr int kAttentionHeads = 16;
constexpr int kEndToken = 151645;
constexpr int kMaxTokens = 4096;
constexpr int kLayers = 28;
constexpr int kHiddenLayerNeurons = 6144;
constexpr int kVocabSize = 151936;
constexpr int ropeTheta = 1000000;

constexpr float kRMSNormEps = 1e-6;

// 28 layers * 8 KV heads per layer -> 224 caches
vector<KVMatrix> key_cache(224, KVMatrix(kMaxTokens, 128));
vector<KVMatrix> value_cache(224, KVMatrix(kMaxTokens, 128));
vector<int> indices(224, 0);

HeapMatrix embeddingTable(kVocabSize, 2048);

vector<RowVec> layerInputNormWeights(28, RowVec(2048));

vector<HeapMatrix> layerQueryWeights(28, HeapMatrix(2048, 2048));
vector<HeapMatrix> layerKeyWeights(28, HeapMatrix(1024, 2048));
vector<HeapMatrix> layerValueWeights(28, HeapMatrix(1024, 2048));

vector<RowVec> layerKeyNormWeights(28, RowVec(128));
vector<RowVec> layerQueryNormWeights(28, RowVec(128));

vector<HeapMatrix> attentionOutputWeights(28, HeapMatrix(2048, 2048));

vector<RowVec> postAttentionNormWeights(28, RowVec(2048));

vector<HeapMatrix> layerGateWeights(28, HeapMatrix(6144, 2048));
vector<HeapMatrix> layerUpWeights(28, HeapMatrix(6144, 2048));
vector<HeapMatrix> layerDownWeights(28, HeapMatrix(2048, 6144));

RowVec modelNormWeights(2048); 


template <typename T>
void read_matrix_params(string param_name, T &matrix) {
    model_loader loader;
    string f_name = "qwen3-1.7B/" + loader.get_file_name(param_name);
    ifstream f(f_name);

    param_offset offset = loader.get_param_offset(param_name);
    uint64_t start = offset.start;
    f.seekg(start);

    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            uint16_t b;
            f.read(reinterpret_cast<char*>(&b), sizeof(b));
            float fl = bit_cast<float>(b << 16);
            matrix(i, j) = fl;
        }
    }
}

void parameterCache() {
    cout << "Caching model params..." << endl;
    // Embedding table for all tokens in vocab
    string token_embedding_params = "model.embed_tokens.weight";
    read_matrix_params(token_embedding_params, embeddingTable);

    for (int layer = 0; layer < kLayers; layer++) {
        // Layer input normalization weights
        string layer_input_norm_params = format("model.layers.{}.input_layernorm.weight", layer);
        read_matrix_params(layer_input_norm_params, layerInputNormWeights[layer]);


        // Collecting query, key, and value matrix weights across layers
        string query_weight_params = format("model.layers.{}.self_attn.q_proj.weight", layer);
        read_matrix_params(query_weight_params, layerQueryWeights[layer]);

        string key_weight_params = format("model.layers.{}.self_attn.k_proj.weight", layer);
        read_matrix_params(key_weight_params, layerKeyWeights[layer]);

        string value_weight_params = format("model.layers.{}.self_attn.v_proj.weight", layer);
        read_matrix_params(value_weight_params, layerValueWeights[layer]);


        // Collecting query and key norm weights
        string query_norm_params = format("model.layers.{}.self_attn.q_norm.weight", layer);
        read_matrix_params(query_norm_params, layerQueryNormWeights[layer]);

        string key_norm_params = format("model.layers.{}.self_attn.k_norm.weight", layer);
        read_matrix_params(key_norm_params, layerKeyNormWeights[layer]);


        // Collecting attention output weights across layers
        string attention_output_params = format("model.layers.{}.self_attn.o_proj.weight", layer);
        read_matrix_params(attention_output_params, attentionOutputWeights[layer]);


        // Post attention normalization weights
        string post_attention_norm_params = format("model.layers.{}.post_attention_layernorm.weight", layer);
        read_matrix_params(post_attention_norm_params, postAttentionNormWeights[layer]);


        // Collecting gate, up, and down weights for all layers
        string gate_weight_params = format("model.layers.{}.mlp.gate_proj.weight", layer);
        read_matrix_params(gate_weight_params, layerGateWeights[layer]);

        string up_weight_params = format("model.layers.{}.mlp.up_proj.weight", layer);
        read_matrix_params(up_weight_params, layerUpWeights[layer]);

        string down_weight_params = format("model.layers.{}.mlp.down_proj.weight", layer);
        read_matrix_params(down_weight_params, layerDownWeights[layer]);
    }

    // Normalization on residual stream after all layers
    string model_norm_params = "model.norm.weight";
    read_matrix_params(model_norm_params, modelNormWeights);
    cout << "Finished caching params" << endl;
}

// Softmax matrix rows in-place
template <typename T>
void softmax(T &matrix) {
    for (int row = 0; row < matrix.rows(); row++) {
        float maxVal = matrix(row, 0);
        for (int j = 1; j < matrix.cols(); j++) {
            maxVal = max(matrix(row, j), maxVal);
        }

        float denom = 0;
        for (int j = 0; j < matrix.cols(); j++) {
            matrix(row, j) -= maxVal;
            denom += exp(matrix(row, j));
        }

        for (int j = 0; j < matrix.cols(); j++) {
            matrix(row, j) = exp(matrix(row, j));
            matrix(row, j) /= denom;
        }
    }
}

float SiLU(float val) {
    return val / (1.0 + exp(-val));
}

template <typename T>
void RoPE(T &&vec, int pos) {
    for (int i = 0; i < 64; i++) {
        float exponent = (float)i / -64.0;
        float theta = pow(ropeTheta, exponent);
        float origI = vec[i];
        vec[i] = (vec[i] * cos(pos * theta) - vec[i + 64] * sin(pos * theta));
        vec[i + 64] = (origI * sin(pos * theta) + vec[i + 64] * cos(pos * theta));
    }
}

Engine::Engine(vector<int> ids) {
    input_ids = ids;
}

// Return a copy of the normalized matrix
HeapMatrix RMS(const RowVec &params, HeapMatrix &matrix) {
    HeapMatrix normed(matrix.rows(), matrix.cols());
    for (int i = 0; i < matrix.rows(); i++) {
        float rms = 0.0f;
        for (int j = 0; j < matrix.cols(); j++) {
            rms += pow(matrix(i, j), 2);
        }

        rms /= matrix.cols();
        rms += kRMSNormEps;
        rms = sqrt(rms);

        for (int j = 0; j < matrix.cols(); j++) {
            normed(i, j) = (matrix(i, j) / rms) * params(j);
        }
    }

    return normed;
} 

HeapMatrix attention(HeapMatrix &stream, int layer) {
    cout << "LAYER " << layer << endl;
    int tokens = stream.rows();

    HeapMatrix queryWeightsT = layerQueryWeights[layer].transpose();
    HeapMatrix keyWeightsT = layerKeyWeights[layer].transpose();
    HeapMatrix valueWeightsT = layerValueWeights[layer].transpose();

    // Query, key, and value matrices for the entire layer
    HeapMatrix queries = stream * queryWeightsT;
    HeapMatrix keys = stream * keyWeightsT;
    HeapMatrix values = stream * valueWeightsT;
    vector<WeightedValueMatrix> weightedVals(16, WeightedValueMatrix::Zero(tokens, 128));

    for (int i = 0; i < kAttentionHeads; i++) {
        HeapMatrix headQueries = queries.block(0, 128 * i, tokens, 128);
        HeapMatrix headKeys = keys.block(0, 128 * (i / 2), tokens, 128);
        HeapMatrix headValues = values.block(0, 128 * (i / 2), tokens, 128);

        RowVec queryNorm = layerQueryNormWeights[layer];
        RowVec keyNorm = layerKeyNormWeights[layer];
        headQueries = RMS(queryNorm, headQueries);
        headKeys = RMS(keyNorm, headKeys);

        int curLayer = 8 * layer + i / 2;
        int prevTokens = indices[curLayer] - (i % 2) * tokens;

        // Apply RoPE to the head's query and key vectors
        for (int i = 0; i < headQueries.rows(); i++) {
            RoPE(headQueries.row(i), prevTokens + i);
            RoPE(headKeys.row(i), prevTokens + i);
        }

        // Appending to KV cache before taking scores (tokens should be able to attend to themselves)
        int index = indices[curLayer];
        int T = (i % 2 == 0) ? tokens : 0;
        for (int j = index; j < index + T; j++) {
            for (int k = 0; k < 128; k++) {
                key_cache[curLayer](j, k) = headKeys(j - index, k);
                value_cache[curLayer](j, k) = headValues(j - index, k);
            }
        }
        indices[curLayer] = index + T;

        // Calculating score matrix
        HeapMatrix S = headQueries * key_cache[curLayer].block(0, 0, indices[curLayer], 128).transpose();

        // Normalize scores, apply causal mask, and then softmax
        for (int row = 0; row < S.rows(); row++) {
            for (int col = 0; col < S.cols(); col++) {
                S(row, col) /= sqrt(128);
                if (col > row + prevTokens) {
                    S(row, col) = INT_MIN;
                }
            }
        }
        softmax(S);

        // Create a sum of weighted value vectors for each token
        weightedVals[i] = S * value_cache[curLayer].topRows(indices[curLayer]);
    }

    // cout << "creating concatenation values matrix" << endl;
    ConcatenatedValueMatrix concatenatedValues(tokens, 2048);
    for (int i = 0; i < kAttentionHeads; i++) {
        for (int j = 0; j < tokens; j++) {
            for (int k = 0; k < 128; k++) {
                concatenatedValues(j, i * 128 + k) = weightedVals[i](j, k);
            }
        }
    }
    
    ConcatenatedValueMatrix concatenatedProjection = concatenatedValues * attentionOutputWeights[layer].transpose();
    return concatenatedProjection;
}

HeapMatrix mlp(HeapMatrix &stream, int layer) {
    MatrixXf gateWeights = layerGateWeights[layer];
    MatrixXf upWeights = layerUpWeights[layer];
    MatrixXf downWeights = layerDownWeights[layer];
    int tokens = stream.rows();

    HeapMatrix output(tokens, 2048);

    for (int i = 0; i < tokens; i++) {
        vector<float> intermediate(6144);

        for (int neuron = 0; neuron < kHiddenLayerNeurons; neuron++) {
            float gate = 0.0;
            float weight = 0.0;

            for (int k = 0; k < 2048; k++) {
                gate += (gateWeights(neuron, k) * stream(i, k));
                weight += (upWeights(neuron, k) * stream(i, k));
            }

            intermediate[neuron] = SiLU(gate) * weight;
        }

        MatrixXf tokenOutput(1, 2048);
        for (int neuron = 0; neuron < 2048; neuron++) {
            float o = 0.0;
            for (int k = 0; k < kHiddenLayerNeurons; k++) {
                o += (intermediate[k] * downWeights(neuron, k));
            }
            tokenOutput(0, neuron) = o;
        }

        output.row(i) = tokenOutput;
    }

    return output;
}

int sampledToken(RowVec &lastToken) {
    RowVec dotProds(kVocabSize);
    for (int i = 0; i < kVocabSize; i++) {
        RowVec vocabEmbedding = embeddingTable.row(i);
        float accumulated = 0.0;
        for (int j = 0; j < 2048; j++) {
            accumulated += lastToken(j) * vocabEmbedding(j);
        }
        dotProds(i) = accumulated;
    }

    // Greedy decoding for simplicity (No need to softmax as softmax is order-preserving)
    int bestNextToken = 0;
    for (int i = 1; i < kVocabSize; i++) {
        if (dotProds(i) > dotProds(bestNextToken)) {
            bestNextToken = i;
        }
    }

    return bestNextToken;
}

vector<int> Engine::forward_pass() {
    parameterCache();

    // Gather token embeddings for the prompt tokens
    MatrixXf tok_embeds(input_ids.size(), 2048);
    for (int i = 0; i < input_ids.size(); i++) {
        RowVec embedding = embeddingTable.row(input_ids[i]);
        for (int j = 0; j < 2048; j++) {
            tok_embeds(i, j) = embedding(j);
        }
    }

    vector<int> generated;
    int outputTokenId = -1;

    HeapMatrix residual = tok_embeds;

    while (outputTokenId != kEndToken) {
        for (int i = 0; i < kLayers; i++) {
            RowVec layerInputNorm = layerInputNormWeights[i];
            HeapMatrix input_normed = RMS(layerInputNorm, residual);
            HeapMatrix attention_values = attention(input_normed, i);

            residual += attention_values;

            RowVec postAttentionNorm = postAttentionNormWeights[i];
            input_normed = RMS(postAttentionNorm, residual);
            MatrixXf mlp_values = mlp(input_normed, i);

            residual += mlp_values;
        }

        residual = RMS(modelNormWeights, residual);

        RowVec finalToken = residual.row(residual.rows() - 1);
        outputTokenId = sampledToken(finalToken);
        generated.push_back(outputTokenId);

        cout << "GENERATED " << outputTokenId << endl;

        residual = HeapMatrix(1, 2048);
        for (int i = 0; i < 2048; i++) {
            residual.row(0) = embeddingTable.row(outputTokenId);
        }
    }   

    return generated;
}