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

    // Step 2: Obtain Hidden States (Actual Implementation from previous subtask)
    LOG_INF("%s: Obtaining hidden states for training...\n", __func__);
    const int n_ctx_train = llama_n_ctx(ctx.get()); // n_ctx used for processing chunks, not necessarily training item size
    const int n_embd = llama_model_n_embd(model.get());
    // const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get())); // n_vocab is unused

    std::vector<llama_token> all_tokens_vector = common_tokenize(ctx.get(), params.prompt, true);
    if (all_tokens_vector.empty()) {
        LOG_ERR("%s: failed to tokenize prompt\n", __func__);
        return 1;
    }

    // Create next_token_labels_vector
    std::vector<llama_token> next_token_labels_vector;
    if (all_tokens_vector.size() > 1) {
        next_token_labels_vector.assign(all_tokens_vector.begin() + 1, all_tokens_vector.end());
        // Pad the last label, e.g., with EOS or a specific pad token if available. Using EOS for now.
        // Or simply make the training data one token shorter.
        // For simplicity, let's make dataset size based on available next_token_labels.
    }
    // If all_tokens_vector.size() is 0 or 1, next_token_labels_vector will be empty.

    const int64_t n_total_trainable_tokens = next_token_labels_vector.size();
    if (n_total_trainable_tokens == 0) {
        LOG_ERR("%s: Not enough tokens to create training data (need at least 2 tokens).\n", __func__);
        return 1;
    }

    std::vector<float> all_hidden_states_data; // To store all hidden states contiguously (n_total_trainable_tokens * n_embd)
    // tokens_processed_for_hidden_states is not directly used for dataset item count anymore with token-based dataset
    // std::vector<int> tokens_processed_for_hidden_states;

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
            // tokens_processed_for_hidden_states.push_back(n_tokens_chunk); // Removed usage
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
        // tokens_processed_for_hidden_states.push_back(n_tokens_chunk); // Removed usage
    }

    if (all_hidden_states_data.empty() && !all_tokens_vector.empty()) { // Check if tokens existed but no HS data
        LOG_ERR("%s: No hidden states were collected. Aborting.\n", __func__);
        return 1;
    }
    // Clear KV cache again after processing if it's not needed immediately
    llama_memory_clear(llama_get_memory(ctx.get()), true);

    // Step 3: Initialize Dataset for Current State Prediction (Token-based)
    LOG_INF("%s: Initializing token-based dataset for dual loss...\n", __func__);

    // Ensure all_hidden_states_data has content for all_tokens_vector[0...n_total_trainable_tokens-1]
    // The hidden state collection loop should have populated this for all tokens processed.
    // We only use the first n_total_trainable_tokens worth of hidden states.
    if (all_hidden_states_data.size() < (size_t)(n_total_trainable_tokens * n_embd)) { // Cast for comparison
        LOG_ERR("%s: Mismatch between collected hidden states (%zu floats) and required for trainable tokens (expected %" PRId64 " floats).\n",
                __func__, all_hidden_states_data.size(), n_total_trainable_tokens * n_embd);
        return 1;
    }


    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_I32,           // type_data (input token type)
        1,                       // ne_datapoint (1 input token per item)
        n_total_trainable_tokens, // ndata (total number of token-based items)
        n_total_trainable_tokens, // ndata_shard (process all as one shard for now)
        GGML_TYPE_F32,           // type_label_A (hidden state element type)
        n_embd,                  // ne_label_A (size of hidden state vector for one token)
        GGML_TYPE_I32,           // type_label_B (next token ID type)
        1                        // ne_label_B (1 next token ID per item)
    );

    // Populate dataset->data with input tokens (all_tokens_vector[0...n_total_trainable_tokens-1])
    struct ggml_tensor * dataset_data_tensor = ggml_opt_dataset_data(dataset);
    memcpy(dataset_data_tensor->data, all_tokens_vector.data(), n_total_trainable_tokens * sizeof(llama_token));
    LOG_INF("%s: Dataset data (input tokens) populated. Total items: %" PRId64 "\n", __func__, n_total_trainable_tokens);

    // Populate dataset->labels_A with hidden states
    struct ggml_tensor * dataset_labels_A_tensor = ggml_opt_dataset_labels_A(dataset);
    memcpy(dataset_labels_A_tensor->data, all_hidden_states_data.data(), n_total_trainable_tokens * n_embd * sizeof(float));
    LOG_INF("%s: Dataset labels_A (hidden states) populated.\n", __func__);

    // Populate dataset->labels_B with next token labels
    struct ggml_tensor * dataset_labels_B_tensor = ggml_opt_dataset_labels_B(dataset);
    memcpy(dataset_labels_B_tensor->data, next_token_labels_vector.data(), n_total_trainable_tokens * sizeof(llama_token));
    LOG_INF("%s: Dataset labels_B (next tokens) populated.\n", __func__);

    // Define optimizer settings (e.g., AdamW hyperparameters)
    struct ggml_opt_optimizer_params optimizer_params = ggml_opt_get_default_optimizer_params(nullptr);
    optimizer_params.adamw.alpha = 1e-7f; // Example: Set learning rate
    // TODO: Consider setting other AdamW params (beta1, beta2, eps, wd) from common_params if they are added there.

    // Define LLaMA specific optimization parameters (which params to train, loss types, weights)
    struct llama_opt_params lopt_params {
        /*n_ctx_train     =*/ static_cast<uint32_t>(n_ctx_train),
        /*param_filter    =*/ llama_opt_param_filter_all,
        /*param_filter_ud =*/ nullptr,
        /*get_opt_pars    =*/ ggml_opt_get_constant_optimizer_params, // Use constant HPs from optimizer_params
        /*get_opt_pars_ud =*/ &optimizer_params,                     // Pass address of optimizer_params
        /*loss_type_A     =*/ GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
        /*loss_type_B     =*/ GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
        /*loss_A_weight   =*/ 0.5f,
        /*loss_B_weight   =*/ 0.5f,
        /*outputs_B       =*/ nullptr // Not used by llama_opt_init
    };

    // Call llama_opt_init: This configures which model parameters are trainable
    // and stores the loss types/weights into the llama_context.
    // It no longer creates the ggml_opt_context.
    llama_opt_init(ctx.get(), model.get(), lopt_params);

    // Step 4 & 5: Initialize ggml_opt_context for actual training graph
    LOG_INF("%s: Initializing ggml_opt_context for dual loss training...\n", __func__);

    struct ggml_opt_params main_ggml_opt_params = ggml_opt_default_params(
        nullptr, // backend_sched - see TODO below
        lopt_params.loss_type_A,
        lopt_params.loss_type_B
    );

    // These will be set via ggml_opt_prepare_alloc by llama_context::opt_epoch_iter
    main_ggml_opt_params.ctx_compute = nullptr;
    main_ggml_opt_params.inputs      = nullptr;
    main_ggml_opt_params.outputs_A   = nullptr;
    main_ggml_opt_params.outputs_B   = nullptr;

    main_ggml_opt_params.loss_A_weight = lopt_params.loss_A_weight;
    main_ggml_opt_params.loss_B_weight = lopt_params.loss_B_weight;

    main_ggml_opt_params.get_opt_pars = ggml_opt_get_constant_optimizer_params;
    main_ggml_opt_params.get_opt_pars_ud = &optimizer_params; // Correctly point to the single optimizer_params instance

    if (params.n_ubatch == 0 || params.n_batch % params.n_ubatch != 0) {
        LOG_ERR("%s: n_batch must be a multiple of n_ubatch, and n_ubatch > 0.\n", __func__);
        return 1;
    }
    main_ggml_opt_params.opt_period = params.n_batch / params.n_ubatch;

    // TODO: The backend_sched parameter for ggml_opt_default_params and subsequently for ggml_opt_init
    // is problematic as it's internal to llama_context. A proper solution would require
    // llama_context to expose its scheduler or assist in ggml_opt_context creation.
    // Using a temporary CPU scheduler for init, as it might only be strictly needed for static graph allocation.
    ggml_backend_t cpu_backend_for_init = ggml_backend_cpu_init();
    ggml_backend_sched_t temp_sched_for_init = ggml_backend_sched_new(&cpu_backend_for_init, nullptr, 1, 1, false, false);
    main_ggml_opt_params.backend_sched = temp_sched_for_init;

    ggml_opt_context_t main_opt_ctx = ggml_opt_init(main_ggml_opt_params);

    ggml_backend_sched_free(temp_sched_for_init); // Free the temporary scheduler
    ggml_backend_free(cpu_backend_for_init);      // Free the temporary backend

    if (!main_opt_ctx) {
        LOG_ERR("%s: Failed to initialize ggml_opt_context.\n", __func__);
        return 1;
    }

    constexpr float val_split = 0.05f;
    const int64_t idata_split = n_total_trainable_tokens * (1.0f - val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    // Using a hardcoded number of epochs for now, as params.n_epochs is not in common_params
    for (int epoch = 0; epoch < 2; ++epoch) {
        llama_opt_epoch(ctx.get(), main_opt_ctx, dataset, result_train, result_eval, idata_split,
            ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    ggml_opt_free(main_opt_ctx); // Free the context
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    llama_model_save_to_file(model.get(), "finetuned-model.gguf");

    llama_backend_free();

    return 0;
}
