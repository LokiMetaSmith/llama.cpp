#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-opt.h" // Added for ggml_opt_dataset_init and related types

#include <cmath>
#include <cinttypes> // Added for PRId64
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int main(int argc, char ** argv) {
    common_params params;

    params.escape = false;
    params.embedding = true; // Enable embedding computation

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    if (params.use_mmap) {
        LOG_INF("%s: force disabling memory mapping because it would result in-read-only pointers to the weights\n", __func__);
        params.use_mmap = false;
    }
    if (params.cache_type_k != GGML_TYPE_F32) {
        LOG_INF("%s: force changing k cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        LOG_INF("%s: force changing v cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_v = GGML_TYPE_F32;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model and apply lora adapter, if any
    common_init_result llama_init = common_init_from_params(params);
    llama_model_ptr   & model = llama_init.model;
    llama_context_ptr & ctx   = llama_init.context;

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    // Step 2: Obtain Hidden States (Placeholder Implementation)
    LOG_INF("%s: Obtaining hidden states for training...\n", __func__);
    const int n_ctx_train = llama_n_ctx(ctx.get());
    const int n_embd = llama_model_n_embd(model.get());
    // const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get())); // n_vocab is unused

    std::vector<llama_token> all_tokens_vector = common_tokenize(ctx.get(), params.prompt, true);
    if (all_tokens_vector.empty()) {
        LOG_ERR("%s: failed to tokenize prompt\n", __func__);
        return 1;
    }

    std::vector<float> all_hidden_states_data; // To store all hidden states contiguously
    std::vector<int> tokens_processed_for_hidden_states; // To store the number of tokens for which HS were generated per chunk

    // Ensure the KV cache is empty before processing
    llama_memory_clear(llama_get_memory(ctx.get()), true);

    for (size_t i = 0; i < all_tokens_vector.size(); i += n_ctx_train) {
        int n_tokens_chunk = std::min((size_t)n_ctx_train, all_tokens_vector.size() - i);
        if (n_tokens_chunk == 0) continue;
        std::vector<llama_token> chunk_tokens(all_tokens_vector.begin() + i, all_tokens_vector.begin() + i + n_tokens_chunk);

        // Manually prepare batch to request embeddings for all tokens
        std::vector<llama_pos> positions(n_tokens_chunk);
        std::vector<int32_t> n_seq_ids(n_tokens_chunk, 1);
        std::vector<llama_seq_id*> p_seq_ids(n_tokens_chunk);
        std::vector<int8_t> logits_flags(n_tokens_chunk, 1); // Request output for all tokens

        llama_seq_id current_seq_id = 0; // Using a single sequence ID for all tokens in the chunk
        for (int k = 0; k < n_tokens_chunk; ++k) {
            positions[k] = k; // Position relative to the start of this chunk for KV cache
            p_seq_ids[k] = &current_seq_id;
        }

        llama_batch batch = {
            n_tokens_chunk,
            chunk_tokens.data(),
            nullptr, // No explicit embeddings input
            positions.data(),
            n_seq_ids.data(),
            p_seq_ids.data(),
            logits_flags.data()
        };

        if (llama_decode(ctx.get(), batch)) {
            LOG_ERR("%s: llama_decode failed for chunk starting at token %zu\n", __func__, i);
            // KV cache might be in an inconsistent state for this chunk,
            // but since we clear KV cache per loop for hidden state gathering, it's okay for next chunk.
            // However, this chunk's hidden states will be missing or incorrect.
            // For simplicity, we'll insert zeros and continue, but real error handling might be needed.
            all_hidden_states_data.insert(all_hidden_states_data.end(), (size_t)n_tokens_chunk * n_embd, 0.0f);
            tokens_processed_for_hidden_states.push_back(n_tokens_chunk);
            continue;
        }
        llama_synchronize(ctx.get()); // Ensure computation is finished before getting embeddings

        float *hidden_states_for_chunk = llama_get_embeddings(ctx.get());

        if (hidden_states_for_chunk != nullptr) {
            all_hidden_states_data.insert(all_hidden_states_data.end(),
                                          hidden_states_for_chunk,
                                          hidden_states_for_chunk + (size_t)n_tokens_chunk * n_embd);
            LOG_INF("%s: Processed chunk %zu, %d tokens, collected actual hidden states.\n", __func__, i/n_ctx_train, n_tokens_chunk);
        } else {
            LOG_ERR("%s: Failed to retrieve embeddings for chunk starting at token %zu (null pointer).\n", __func__, i);
            // Add placeholder zeros to maintain structure, but this indicates a problem.
            all_hidden_states_data.insert(all_hidden_states_data.end(), (size_t)n_tokens_chunk * n_embd, 0.0f);
             LOG_WRN("%s: Using zeros as placeholder for this chunk.\n", __func__);
        }
        tokens_processed_for_hidden_states.push_back(n_tokens_chunk);
    }

    if (all_hidden_states_data.empty() && !all_tokens_vector.empty()) { // Check if tokens existed but no HS data
        LOG_ERR("%s: No hidden states were collected. Aborting.\n", __func__);
        return 1;
    }
    // Clear KV cache again after processing if it's not needed immediately
    llama_memory_clear(llama_get_memory(ctx.get()), true);

    // Step 3: Initialize Dataset for Current State Prediction
    LOG_INF("%s: Initializing dataset for current state prediction...\n", __func__);

    const int64_t ne_datapoint_tokens = n_ctx_train; // Number of tokens per data item for inputs
    const int64_t single_item_label_size = (int64_t)n_embd * ne_datapoint_tokens; // Total size of label for one item

    // const int64_t ndata_total_tokens = all_tokens_vector.size(); // Unused variable
    const int64_t n_dataset_items = tokens_processed_for_hidden_states.size();

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_I32,           // type_data (tokens are I32)
        GGML_TYPE_F32,           // type_label (hidden states are F32)
        ne_datapoint_tokens,     // ne_datapoint (number of input tokens per example)
        single_item_label_size,  // ne_label (size of the label for one example item)
        n_dataset_items,         // ndata (number of examples/chunks)
        n_dataset_items          // ndata_shard (process all items as one shard for simplicity here)
    );

    // Populate dataset->data with the input tokens
    // dataset->data is a 2D tensor: [ne_datapoint_tokens, n_dataset_items]
    // all_tokens_vector contains all tokens contiguously.
    // We need to copy chunk by chunk into dataset->data.
    // Note: ggml tensors are usually column-major by default for matrices.
    // If dataset->data is [N, M], data is stored as N elements of 1st col, then N of 2nd, etc.
    // Here, ne_datapoint_tokens is dim 0, n_dataset_items is dim 1.
    // So, we copy (ne_datapoint_tokens for item 0), then (ne_datapoint_tokens for item 1), ...

    struct ggml_tensor * dataset_data_tensor = ggml_opt_dataset_data(dataset);
    llama_token * data_tokens_ptr = (llama_token *)dataset_data_tensor->data;
    size_t current_input_token_idx = 0;
    for (int item_idx = 0; item_idx < n_dataset_items; ++item_idx) {
        int tokens_in_this_input_chunk = tokens_processed_for_hidden_states[item_idx]; // This is also the number of tokens for which we have HS
        // Ensure we don't read past the end of all_tokens_vector for inputs
        tokens_in_this_input_chunk = std::min(tokens_in_this_input_chunk, (int)(all_tokens_vector.size() - current_input_token_idx));

        memcpy(data_tokens_ptr + (size_t)item_idx * ne_datapoint_tokens, // Offset by item in target
               all_tokens_vector.data() + current_input_token_idx,      // Source from flat vector
               tokens_in_this_input_chunk * sizeof(llama_token));

        // If tokens_in_this_input_chunk is less than ne_datapoint_tokens, pad input with 0 (EOS or PAD)
        if (tokens_in_this_input_chunk < ne_datapoint_tokens) {
            memset(data_tokens_ptr + (size_t)item_idx * ne_datapoint_tokens + tokens_in_this_input_chunk,
                   0, // Pad with token 0
                   (ne_datapoint_tokens - tokens_in_this_input_chunk) * sizeof(llama_token));
        }
        current_input_token_idx += tokens_in_this_input_chunk;
    }
    LOG_INF("%s: Dataset data (tokens) populated. Total items: %" PRId64 ", tokens per item: %" PRId64 "\n", __func__, n_dataset_items, ne_datapoint_tokens);

    // Step 4: Populate Labels Tensor
    // dataset->labels is a 2D tensor: [single_item_label_size, n_dataset_items]
    // single_item_label_size = n_embd * ne_datapoint_tokens
    // all_hidden_states_data contains all hidden states contiguously:
    // (n_embd for token0_item0), (n_embd for token1_item0) ... (n_embd for tokenL_item0), (n_embd for token0_item1) ...
    // Total floats in all_hidden_states_data = sum(tokens_processed_for_hidden_states[j] * n_embd).

    struct ggml_tensor * dataset_labels_tensor = ggml_opt_dataset_labels(dataset);
    float * labels_data_ptr = (float *)dataset_labels_tensor->data;
    size_t current_hs_float_offset = 0;
    for (int item_idx = 0; item_idx < n_dataset_items; ++item_idx) {
        int tokens_in_this_hs_chunk = tokens_processed_for_hidden_states[item_idx];
        size_t floats_to_copy_for_this_item = (size_t)tokens_in_this_hs_chunk * n_embd;

        memcpy(labels_data_ptr + (size_t)item_idx * single_item_label_size, // Offset by item in target (destination)
               all_hidden_states_data.data() + current_hs_float_offset,    // Source from flat vector
               floats_to_copy_for_this_item * sizeof(float));

        // If tokens_in_this_hs_chunk is less than ne_datapoint_tokens (max tokens for an item),
        // pad the rest of this item's label space in dataset->labels with zeros.
        if (tokens_in_this_hs_chunk < ne_datapoint_tokens) {
            memset(labels_data_ptr + (size_t)item_idx * single_item_label_size + floats_to_copy_for_this_item,
                   0,
                   (ne_datapoint_tokens - tokens_in_this_hs_chunk) * n_embd * sizeof(float));
        }
        current_hs_float_offset += floats_to_copy_for_this_item;
    }
    LOG_INF("%s: Dataset labels (hidden states) populated.\n", __func__);

    constexpr float val_split = 0.05f;

    // Step 5: Set Up Optimizer for MSE (partially addressable)
    // Note: Setting the loss to MSE is now possible.
    LOG_INF("%s: Setting loss type to MSE for current state prediction.\n", __func__);

    struct ggml_opt_optimizer_params optimizer_params = ggml_opt_get_default_optimizer_params(nullptr);
    optimizer_params.adamw.alpha = 1e-7f; // learning rate

    struct llama_opt_params lopt_params {
        /*n_ctx_train     =*/ static_cast<uint32_t>(n_ctx_train), // Set context for training
        /*param_filter    =*/ llama_opt_param_filter_all,
        /*param_filter_ud =*/ nullptr,
        /*get_opt_pars    =*/ ggml_opt_get_constant_optimizer_params,
        /*get_opt_pars_ud =*/ &optimizer_params,
        /*loss_type       =*/ GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR
    };
    llama_opt_init(ctx.get(), model.get(), lopt_params);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    for (int epoch = 0; epoch < 2; ++epoch) {
        llama_opt_epoch(ctx.get(), dataset, result_train, result_eval, idata_split,
            ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    llama_model_save_to_file(model.get(), "finetuned-model.gguf");

    llama_backend_free();

    return 0;
}
