#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-opt.h" // Added for ggml_opt_dataset_init and related types

#include <cmath>
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
    const int n_embd = llama_n_embd(model.get());
    const int n_vocab = llama_n_vocab(model.get());

    std::vector<llama_token> all_tokens_vector = common_tokenize(ctx.get(), params.prompt, true);
    if (all_tokens_vector.empty()) {
        LOG_ERR("%s: failed to tokenize prompt\n", __func__);
        return 1;
    }

    std::vector<float> all_hidden_states_data; // To store all hidden states contiguously
    std::vector<int> tokens_processed_for_hidden_states; // To store the number of tokens for which HS were generated per chunk

    // Ensure the KV cache is empty before processing
    llama_kv_cache_clear(ctx.get());

    for (size_t i = 0; i < all_tokens_vector.size(); i += n_ctx_train) {
        int n_tokens_chunk = std::min((size_t)n_ctx_train, all_tokens_vector.size() - i);
        std::vector<llama_token> chunk_tokens(all_tokens_vector.begin() + i, all_tokens_vector.begin() + i + n_tokens_chunk);

        // Process the chunk. llama_decode updates the KV cache and computes logits.
        // We need the hidden states *before* the final projection to logits.
        if (llama_decode(ctx.get(), llama_batch_get_one(chunk_tokens.data(), n_tokens_chunk, 0, 0))) {
            LOG_ERR("%s: llama_decode failed for chunk starting at token %zu\n", __func__, i);
            // KV cache might be in an inconsistent state, clear it or handle error
            llama_kv_cache_clear(ctx.get());
            continue;
        }

        // TODO: Replace this placeholder with actual hidden state extraction.
        // This currently extracts logits, not hidden states. We need the tensor *before*
        // the final llama_model.output linear layer (output_norm -> dense -> output).
        // For now, as a placeholder, let's assume hidden states have n_embd dimension
        // and are F32. We will fill with zeros.
        // The actual hidden states would be of shape [n_tokens_chunk, n_embd].

        // Placeholder: Get "embeddings" which are often the direct output of the transformer layers.
        // Or, if llama_get_logits gives per-token logits, its shape would be [n_tokens_chunk, n_vocab].
        // We need a tensor of shape [n_tokens_chunk, n_embd].
        // For this placeholder, we'll just allocate zeros of the correct size.
        std::vector<float> current_chunk_hidden_states(n_tokens_chunk * n_embd, 0.0f);

        // --- This is where the actual hidden state extraction logic would go ---
        // Example of what might be needed if a specific tensor is accessible:
        // struct ggml_tensor * last_hidden_state_t = llama_get_tensor_by_name(ctx.get(), "model.layers.N-1.output"); // Fictional function
        // if (last_hidden_state_t && last_hidden_state_t->type == GGML_TYPE_F32) {
        //     GGML_ASSERT(ggml_nelements(last_hidden_state_t) == n_tokens_chunk * n_embd);
        //     memcpy(current_chunk_hidden_states.data(), (float*)last_hidden_state_t->data, ggml_nbytes(last_hidden_state_t));
        // } else {
        //     LOG_WRN("%s: Could not get actual last hidden state tensor. Using zeros as placeholder.\n", __func__);
        // }
        // --- End of placeholder guidance ---

        all_hidden_states_data.insert(all_hidden_states_data.end(), current_chunk_hidden_states.begin(), current_chunk_hidden_states.end());
        tokens_processed_for_hidden_states.push_back(n_tokens_chunk);
        LOG_INF("%s: Processed chunk %zu, %d tokens, collected placeholder hidden states.\n", __func__, i/n_ctx_train, n_tokens_chunk);
    }

    if (all_hidden_states_data.empty()) {
        LOG_ERR("%s: No hidden states were collected. Aborting.\n", __func__);
        return 1;
    }
    // Clear KV cache again after processing if it's not needed immediately
    llama_kv_cache_clear(ctx.get());

    // Step 3: Initialize Dataset for Current State Prediction
    LOG_INF("%s: Initializing dataset for current state prediction...\n", __func__);

    // Create an exemplar hidden state tensor
    // This is a dummy tensor to describe the structure of the labels.
    // For sequence-to-sequence, if each item has 'ne_datapoint_tokens' and each token's label is 'n_embd' floats,
    // then one item's label is 'ne_datapoint_tokens * n_embd' floats.
    // hidden_state_exemplar->ne[0] should be this size.
    const int64_t ne_datapoint_tokens = n_ctx_train; // Number of tokens per data item for inputs
    const int64_t single_item_label_size = n_embd * ne_datapoint_tokens;

    struct ggml_init_params exemplar_params = {
        .mem_size   = ggml_tensor_overhead(GGML_TYPE_F32, GGML_MAX_DIMS), // Minimal size
        .mem_buffer = nullptr,
        .no_alloc   = true, // Don't allocate data for it
    };
    struct ggml_context * ctx_exemplar = ggml_init(exemplar_params);
    struct ggml_tensor * hidden_state_exemplar = ggml_new_tensor_1d(ctx_exemplar, GGML_TYPE_F32, single_item_label_size);

    const int64_t ndata_total_tokens = all_tokens_vector.size();
    // We need to ensure that ndata for dataset->data (tokens) and dataset->labels (hidden_states) align.
    // Each "item" in the dataset corresponds to n_ctx_train tokens and their n_ctx_train * n_embd hidden states.
    // The number of such items (ndata for ggml_opt_dataset_init) is the number of chunks.
    const int64_t n_dataset_items = tokens_processed_for_hidden_states.size();

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_I32,          // type_data (tokens are I32)
        hidden_state_exemplar,  // hidden_state_exemplar for labels
        ne_datapoint_tokens,    // ne_datapoint (number of input tokens per example)
        n_dataset_items,        // ndata (number of examples/chunks)
        n_dataset_items         // ndata_shard (process all items as one shard for simplicity here)
    );
    ggml_free(ctx_exemplar); // Free the temporary context for the exemplar

    // Populate dataset->data with the input tokens
    // dataset->data is a 2D tensor: [ne_datapoint_tokens, n_dataset_items]
    // all_tokens_vector contains all tokens contiguously.
    // We need to copy chunk by chunk into dataset->data.
    // Note: ggml tensors are usually column-major by default for matrices.
    // If dataset->data is [N, M], data is stored as N elements of 1st col, then N of 2nd, etc.
    // Here, ne_datapoint_tokens is dim 0, n_dataset_items is dim 1.
    // So, we copy (ne_datapoint_tokens for item 0), then (ne_datapoint_tokens for item 1), ...

    llama_token * data_tokens_ptr = (llama_token *)dataset->data->data;
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

    float * labels_data_ptr = (float *)dataset->labels->data;
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
        /*n_ctx_train     =*/ n_ctx_train, // Set context for training
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
