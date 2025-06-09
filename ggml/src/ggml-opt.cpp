#include "ggml-opt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cinttypes>
#include <map>
#include <random>
#include <vector>

struct ggml_opt_dataset {
    struct ggml_context   * ctx        = nullptr;
    ggml_backend_buffer_t   buf        = nullptr;
    struct ggml_tensor    * data       = nullptr;
    struct ggml_tensor    * labels_A   = nullptr; // Renamed
    struct ggml_tensor    * labels_B   = nullptr; // Added

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_data    = -1;
    size_t  nbs_labels_A = 0; // Renamed and initialized
    size_t  nbs_labels_B = 0; // Added and initialized

    std::vector<int64_t> permutation;
};

struct ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = nullptr;
    ggml_cgraph              * allocated_graph      = nullptr;
    ggml_cgraph              * allocated_graph_copy = nullptr;
    struct ggml_context      * ctx_static           = nullptr;
    struct ggml_context      * ctx_cpu              = nullptr;
    struct ggml_context      * ctx_compute          = nullptr;
    struct ggml_context      * ctx_copy             = nullptr;
    ggml_backend_buffer_t      buf_static           = nullptr;
    ggml_backend_buffer_t      buf_cpu              = nullptr;
    std::mt19937               rng;
    // enum ggml_opt_loss_type    loss_type; // Replaced by loss_type_A and loss_type_B
    enum ggml_opt_build_type   build_type;
    enum ggml_opt_build_type   build_type_alloc;

    struct ggml_tensor * inputs    = nullptr;
    struct ggml_tensor * outputs_A = nullptr; // Renamed
    struct ggml_tensor * outputs_B = nullptr; // Added
    struct ggml_tensor * labels_A  = nullptr; // Renamed
    struct ggml_tensor * labels_B  = nullptr; // Added

    enum ggml_opt_loss_type loss_type_A; // Added
    enum ggml_opt_loss_type loss_type_B; // Added
    float loss_A_weight = 1.0f;          // Added
    float loss_B_weight = 1.0f;          // Added

    struct ggml_tensor * loss     = nullptr; // Combined loss
    struct ggml_tensor * pred     = nullptr; // Associated with outputs_B
    struct ggml_tensor * ncorrect = nullptr; // Associated with outputs_B

    struct ggml_cgraph * gf      = nullptr;
    struct ggml_cgraph * gb_grad = nullptr;
    struct ggml_cgraph * gb_opt  = nullptr;
    bool static_graphs           = false;
    bool eval_ready              = false;
    std::vector<struct ggml_tensor *> grad_accs;
    std::vector<struct ggml_tensor *> grad_m;
    std::vector<struct ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    bool    loss_per_datapoint = false;

    ggml_opt_get_optimizer_params get_opt_pars = nullptr;
    void * get_opt_pars_ud                     = nullptr;
    struct ggml_tensor * adamw_params          = nullptr;
};

struct ggml_opt_result {
    int64_t              ndata    = 0;
    std::vector<float>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    bool    loss_per_datapoint = false;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        enum ggml_type type_data,
        int64_t        ne_datapoint,
        int64_t        ndata,
        int64_t        ndata_shard,
        enum ggml_type type_label_A,
        int64_t        ne_label_A,
        enum ggml_type type_label_B,
        int64_t        ne_label_B) {
    GGML_ASSERT(ne_datapoint > 0);
    GGML_ASSERT(ndata > 0);
    GGML_ASSERT(ndata_shard > 0);
    GGML_ASSERT(ne_label_A >= 0);
    GGML_ASSERT(ne_label_B >= 0);

    ggml_opt_dataset_t result = new ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 3 * ggml_tensor_overhead(), // For data, labels_A, labels_B
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = ggml_init(params);
    }

    result->data = ggml_new_tensor_2d(result->ctx, type_data, ne_datapoint, ndata);
    result->nbs_data = ggml_nbytes(result->data) * ndata_shard / ndata;

    if (ne_label_A > 0) {
        result->labels_A = ggml_new_tensor_2d(result->ctx, type_label_A, ne_label_A, ndata);
        result->nbs_labels_A = ggml_nbytes(result->labels_A) * ndata_shard / ndata;
    } else {
        result->labels_A = nullptr;
        result->nbs_labels_A = 0;
    }

    if (ne_label_B > 0) {
        result->labels_B = ggml_new_tensor_2d(result->ctx, type_label_B, ne_label_B, ndata);
        result->nbs_labels_B = ggml_nbytes(result->labels_B) * ndata_shard / ndata;
    } else {
        result->labels_B = nullptr;
        result->nbs_labels_B = 0;
    }

    result->buf = ggml_backend_alloc_ctx_tensors_from_buft(result->ctx, ggml_backend_cpu_buffer_type());

    const int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    return result;
}

void ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_buffer_free(dataset->buf);
    ggml_free(dataset->ctx);
    delete dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    return dataset->ndata;
}

struct ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset) {
    return dataset->data;
}

struct ggml_tensor * ggml_opt_dataset_labels_A(ggml_opt_dataset_t dataset) { // Renamed
    return dataset->labels_A;
}

struct ggml_tensor * ggml_opt_dataset_labels_B(ggml_opt_dataset_t dataset) { // Added
    return dataset->labels_B;
}

void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    if (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        return;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    const int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

void ggml_opt_dataset_get_batch(
        ggml_opt_dataset_t   dataset,
        struct ggml_tensor * data_batch,
        struct ggml_tensor * labels_A_batch,
        struct ggml_tensor * labels_B_batch,
        int64_t              ibatch) {
    GGML_ASSERT(  data_batch && ggml_is_contiguous(data_batch));
    GGML_ASSERT(!labels_A_batch || ggml_is_contiguous(labels_A_batch));
    GGML_ASSERT(!labels_B_batch || ggml_is_contiguous(labels_B_batch));
    GGML_ASSERT((labels_A_batch == nullptr) == (dataset->labels_A == nullptr));
    GGML_ASSERT((labels_B_batch == nullptr) == (dataset->labels_B == nullptr));
    GGML_ASSERT(                  data_batch->type == dataset->data->type);
    GGML_ASSERT(!labels_A_batch || labels_A_batch->type == dataset->labels_A->type);
    GGML_ASSERT(!labels_B_batch || labels_B_batch->type == dataset->labels_B->type);

    const size_t nb_data_batch = ggml_nbytes(data_batch);
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    if (labels_A_batch) {
        const size_t nb_labels_A_batch = ggml_nbytes(labels_A_batch);
        GGML_ASSERT(nb_labels_A_batch == shards_per_batch * dataset->nbs_labels_A);
    }
    if (labels_B_batch) {
        const size_t nb_labels_B_batch = ggml_nbytes(labels_B_batch);
        GGML_ASSERT(nb_labels_B_batch == shards_per_batch * dataset->nbs_labels_B);
    }

    GGML_ASSERT((ibatch + 1) * shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch * shards_per_batch + ishard_batch];

        const char * ptr_data = (const char *)dataset->data->data + ishard * dataset->nbs_data;
        ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch * dataset->nbs_data, dataset->nbs_data);

        if (labels_A_batch) {
            const char * ptr_labels_A = (const char *)dataset->labels_A->data + ishard * dataset->nbs_labels_A;
            ggml_backend_tensor_set(labels_A_batch, ptr_labels_A, ishard_batch * dataset->nbs_labels_A, dataset->nbs_labels_A);
        }

        if (labels_B_batch) {
            const char * ptr_labels_B = (const char *)dataset->labels_B->data + ishard * dataset->nbs_labels_B;
            ggml_backend_tensor_set(labels_B_batch, ptr_labels_B, ishard_batch * dataset->nbs_labels_B, dataset->nbs_labels_B);
        }
    }
}

void ggml_opt_dataset_get_batch_host(
        ggml_opt_dataset_t dataset,
        void *             data_batch_host,
        size_t             nb_data_batch,
        void *             labels_A_batch_host,
        void *             labels_B_batch_host,
        int64_t            ibatch) {
    GGML_ASSERT((labels_A_batch_host == nullptr) == (dataset->labels_A == nullptr));
    GGML_ASSERT((labels_B_batch_host == nullptr) == (dataset->labels_B == nullptr));
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);

    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    GGML_ASSERT((ibatch + 1) * shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch * shards_per_batch + ishard_batch];

        const char * ptr_data_src         = (const char *)dataset->data->data + ishard * dataset->nbs_data;
        char *       ptr_data_batch_dst   = (char *)data_batch_host + ishard_batch * dataset->nbs_data;
        memcpy(ptr_data_batch_dst, ptr_data_src, dataset->nbs_data);

        if (labels_A_batch_host) {
            const char * ptr_labels_A_src       = (const char *)dataset->labels_A->data + ishard * dataset->nbs_labels_A;
            char *       ptr_labels_A_batch_dst = (char *)labels_A_batch_host + ishard_batch * dataset->nbs_labels_A;
            memcpy(ptr_labels_A_batch_dst, ptr_labels_A_src, dataset->nbs_labels_A);
        }

        if (labels_B_batch_host) {
            const char * ptr_labels_B_src       = (const char *)dataset->labels_B->data + ishard * dataset->nbs_labels_B;
            char *       ptr_labels_B_batch_dst = (char *)labels_B_batch_host + ishard_batch * dataset->nbs_labels_B;
            memcpy(ptr_labels_B_batch_dst, ptr_labels_B_src, dataset->nbs_labels_B);
        }
    }
}

// ====== Model / Context ======

struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata) {
    GGML_UNUSED(userdata);

    ggml_opt_optimizer_params result;

    result.adamw.alpha = 0.001f;
    result.adamw.beta1 = 0.9f;
    result.adamw.beta2 = 0.999f;
    result.adamw.eps   = 1e-8f;
    result.adamw.wd    = 0.0f;

    return result;
}

struct ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void * userdata) {
    return *((struct ggml_opt_optimizer_params *) userdata);
}

struct ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        enum ggml_opt_loss_type   loss_type_A_default,
        enum ggml_opt_loss_type   loss_type_B_default) {
    return {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ nullptr,
        /*inputs          =*/ nullptr,
        /*outputs_A       =*/ nullptr, // Renamed from outputs/logits
        /*outputs_B       =*/ nullptr, // Added
        /*loss_type_A     =*/ loss_type_A_default, // Renamed
        /*loss_type_B     =*/ loss_type_B_default, // Added
        /*loss_A_weight   =*/ 1.0f,    // Added
        /*loss_B_weight   =*/ 1.0f,    // Added (default to 1.0, user can set to 0.0 if not used)
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ nullptr,
    };
}

static ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }

    if (tensor_map.find(tensor) != tensor_map.end()) {
        return tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, sizeof(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->data = tensor->data;
    new_tensor->buffer = tensor->buffer;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    return new_tensor;
}

static ggml_cgraph * dup_graph(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_graph_custom(ctx, src->size, /*grads =*/ true);

    for (int i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    for (int i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    for (int i = 0; i < src->n_nodes; ++i) {
        const size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
        const size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    return dst;
}

static void ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->data) && "when using static graphs the inputs must be allocated statically");

    const bool accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    ggml_set_input(opt_ctx->inputs);
    if (opt_ctx->outputs_A) ggml_set_output(opt_ctx->outputs_A);
    if (opt_ctx->outputs_B) ggml_set_output(opt_ctx->outputs_B);

    int n_param = 0;
    for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        const struct ggml_tensor * node = opt_ctx->gf->nodes[i];
        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(node->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    if (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        const size_t tensors_per_param = (accumulate ? 1 : 0) +
            (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT ? 2 : 0);
        const size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        const size_t size_meta = (n_loss + tensors_per_param*n_param + tensors_const) * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        const size_t size_meta = 1 * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_buffer_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = nullptr;
    }

    struct ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;
    struct ggml_tensor * loss_A_tensor = nullptr;
    struct ggml_tensor * loss_B_tensor = nullptr;

    // Calculate Loss A
    if (opt_ctx->outputs_A && opt_ctx->loss_A_weight > 0.0f) {
        opt_ctx->labels_A = ggml_dup_tensor(ctx_results, opt_ctx->outputs_A);
        ggml_set_input(opt_ctx->labels_A);
        ggml_set_name(opt_ctx->labels_A, "labels_A");

        switch (opt_ctx->loss_type_A) {
            case GGML_OPT_LOSS_TYPE_MEAN: {
                loss_A_tensor = ggml_sum(ctx_results, opt_ctx->outputs_A);
                const float scale_A = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs_A));
                loss_A_tensor = ggml_scale(ctx_results, loss_A_tensor, scale_A);
                ggml_set_name(loss_A_tensor, "loss_A_mean");
                // opt_ctx->loss_per_datapoint will be determined by loss_B if present, or this if only loss_A
                break;
            }
            case GGML_OPT_LOSS_TYPE_SUM: {
                loss_A_tensor = ggml_sum(ctx_results, opt_ctx->outputs_A);
                ggml_set_name(loss_A_tensor, "loss_A_sum");
                break;
            }
            case GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: { // Should not happen for hidden states typically
                loss_A_tensor = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs_A, opt_ctx->labels_A);
                ggml_set_name(loss_A_tensor, "loss_A_ce");
                if (opt_ctx->opt_period > 1) {
                    loss_A_tensor = ggml_scale(ctx_results, loss_A_tensor, 1.0f / opt_ctx->opt_period);
                }
                break;
            }
            case GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
                struct ggml_tensor * error_A = ggml_sub(ctx_results, opt_ctx->outputs_A, opt_ctx->labels_A);
                loss_A_tensor = ggml_sqr(ctx_results, error_A);
                loss_A_tensor = ggml_sum(ctx_results, loss_A_tensor);
                const float scale_A = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs_A));
                loss_A_tensor = ggml_scale(ctx_results, loss_A_tensor, scale_A);
                ggml_set_name(loss_A_tensor, "loss_A_mse");
                break;
            }
        }
        if (loss_A_tensor) {
            loss_A_tensor = ggml_scale(ctx_results, loss_A_tensor, opt_ctx->loss_A_weight);
            ggml_set_name(loss_A_tensor, "loss_A_weighted");
        }
    }

    // Calculate Loss B (typically for next token prediction)
    if (opt_ctx->outputs_B && opt_ctx->loss_B_weight > 0.0f) {
        // For Cross Entropy, labels might need different dimensions than outputs (e.g. I32 for token IDs)
        // However, ggml_dup_tensor copies shape and type.
        // The current ggml_cross_entropy_loss expects labels to be same shape as logits for one-hot like encoding,
        // or it's handled internally if labels are token IDs. Assuming for now that the setup matches ggml's expectations.
        opt_ctx->labels_B = ggml_dup_tensor(ctx_results, opt_ctx->outputs_B);
        ggml_set_input(opt_ctx->labels_B);
        ggml_set_name(opt_ctx->labels_B, "labels_B");

        switch (opt_ctx->loss_type_B) {
            case GGML_OPT_LOSS_TYPE_MEAN: { // Less common for logits
                loss_B_tensor = ggml_sum(ctx_results, opt_ctx->outputs_B);
                const float scale_B = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs_B));
                loss_B_tensor = ggml_scale(ctx_results, loss_B_tensor, scale_B);
                ggml_set_name(loss_B_tensor, "loss_B_mean");
                opt_ctx->loss_per_datapoint = true;
                break;
            }
            case GGML_OPT_LOSS_TYPE_SUM: { // Less common for logits
                loss_B_tensor = ggml_sum(ctx_results, opt_ctx->outputs_B);
                ggml_set_name(loss_B_tensor, "loss_B_sum");
                opt_ctx->loss_per_datapoint = false;
                break;
            }
            case GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
                loss_B_tensor = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs_B, opt_ctx->labels_B);
                ggml_set_name(loss_B_tensor, "loss_B_ce");
                if (opt_ctx->opt_period > 1) {
                    loss_B_tensor = ggml_scale(ctx_results, loss_B_tensor, 1.0f / opt_ctx->opt_period);
                }
                opt_ctx->loss_per_datapoint = true; // CE loss is typically per datapoint (token)
                break;
            }
            case GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: { // Less common for logits
                struct ggml_tensor * error_B = ggml_sub(ctx_results, opt_ctx->outputs_B, opt_ctx->labels_B);
                loss_B_tensor = ggml_sqr(ctx_results, error_B);
                loss_B_tensor = ggml_sum(ctx_results, loss_B_tensor);
                const float scale_B = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs_B));
                loss_B_tensor = ggml_scale(ctx_results, loss_B_tensor, scale_B);
                ggml_set_name(loss_B_tensor, "loss_B_mse");
                opt_ctx->loss_per_datapoint = true;
                break;
            }
        }
        if (loss_B_tensor) {
            loss_B_tensor = ggml_scale(ctx_results, loss_B_tensor, opt_ctx->loss_B_weight);
            ggml_set_name(loss_B_tensor, "loss_B_weighted");
        }
    }

    // Combine losses
    if (loss_A_tensor && loss_B_tensor) {
        opt_ctx->loss = ggml_add(ctx_results, loss_A_tensor, loss_B_tensor);
        ggml_set_name(opt_ctx->loss, "loss_total");
    } else if (loss_A_tensor) {
        opt_ctx->loss = loss_A_tensor;
         // If only loss_A is active, its per_datapoint nature depends on its type (MSE and MEAN are true, SUM is false)
        if (opt_ctx->loss_type_A == GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR || opt_ctx->loss_type_A == GGML_OPT_LOSS_TYPE_MEAN) {
            opt_ctx->loss_per_datapoint = true;
        } else {
            opt_ctx->loss_per_datapoint = false;
        }
    } else if (loss_B_tensor) {
        opt_ctx->loss = loss_B_tensor;
        // loss_per_datapoint is already set by loss_B logic
    } else {
        // No valid loss calculated, create a zero loss to prevent graph errors
        // This case should ideally be handled by ensuring at least one loss weight > 0
        opt_ctx->loss = ggml_new_tensor_1d(ctx_results, GGML_TYPE_F32, 1);
        ggml_set_name(opt_ctx->loss, "loss_zero");
        float zero_value = 0.0f;
        ggml_backend_tensor_set(opt_ctx->loss, &zero_value, 0, sizeof(float));
        opt_ctx->loss_per_datapoint = false;
    }

    ggml_set_output(opt_ctx->loss);
    ggml_set_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    // Pred and NCorrect are associated with outputs_B (typically classification/next-token prediction)
    if (opt_ctx->outputs_B && opt_ctx->loss_type_B == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY && opt_ctx->loss_B_weight > 0.0f) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs_B);
        ggml_set_name(opt_ctx->pred, "pred_B");
        ggml_set_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_count_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels_B));
        ggml_set_name(opt_ctx->ncorrect, "ncorrect_B");
        ggml_set_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }


    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        return;
    }

    if (opt_ctx->grad_accs.empty()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        const int n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor * node = opt_ctx->gf->nodes[i];
            if ((accumulate && (node->flags & GGML_TENSOR_FLAG_PARAM)) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            } else {
                opt_ctx->grad_accs[i] = nullptr;
            }
        }

        if (opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            for (int i = 0; i < n_nodes; ++i) {
                ggml_tensor * node = opt_ctx->gf->nodes[i];
                if (node->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                } else {
                    opt_ctx->grad_m[i] = nullptr;
                    opt_ctx->grad_v[i] = nullptr;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ true);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.data());

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ true);

    opt_ctx->adamw_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, 7);
    ggml_set_input(opt_ctx->adamw_params);
    ggml_set_name(opt_ctx->adamw_params, "adamw_params");

    for (int i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        struct ggml_tensor * node = opt_ctx->gb_opt->nodes[i];
        struct ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_opt, node);

        if (grad && (node->flags & GGML_TENSOR_FLAG_PARAM)) {
            struct ggml_tensor * m        = opt_ctx->grad_m[i];
            struct ggml_tensor * v        = opt_ctx->grad_v[i];
            struct ggml_tensor * opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, node, grad, m, v, opt_ctx->adamw_params);

            ggml_set_name(m,        (std::string("AdamW m for ")    + std::string(node->name)).c_str());
            ggml_set_name(v,        (std::string("AdamW v for ")    + std::string(node->name)).c_str());
            ggml_set_name(opt_step, (std::string("AdamW step for ") + std::string(node->name)).c_str());

            ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
        }
    }

    if (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_buffer_type());
}

ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params) {
    ggml_opt_context_t result = new struct ggml_opt_context;
    result->backend_sched    = params.backend_sched;
    result->ctx_compute      = params.ctx_compute;
    // result->loss_type        = params.loss_type; // Removed
    result->loss_type_A      = params.loss_type_A; // Added
    result->loss_type_B      = params.loss_type_B; // Added
    result->loss_A_weight    = params.loss_A_weight; // Added
    result->loss_B_weight    = params.loss_B_weight; // Added
    result->build_type       = params.build_type;
    result->build_type_alloc = params.build_type;
    result->inputs           = params.inputs;
    result->outputs_A        = params.outputs_A; // Renamed
    result->outputs_B        = params.outputs_B; // Added
    result->opt_period       = params.opt_period;
    result->get_opt_pars     = params.get_opt_pars;
    result->get_opt_pars_ud  = params.get_opt_pars_ud;

    GGML_ASSERT(result->opt_period >= 1);

    result->static_graphs = result->ctx_compute;

    if (!result->static_graphs) {
        GGML_ASSERT(!result->inputs);
        GGML_ASSERT(!result->outputs_A);
        // outputs_B can be null even if static_graphs is true, if only one loss is used.
        // GGML_ASSERT(!result->outputs_B);
        return result;
    }

    GGML_ASSERT(result->inputs);
    // outputs_A must exist if static graphs are used and inputs exist.
    GGML_ASSERT(result->outputs_A);

    result->gf = ggml_new_graph_custom(result->ctx_compute, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.
    ggml_build_forward_expand(result->gf, result->outputs_A);
    if (result->outputs_B) {
        ggml_build_forward_expand(result->gf, result->outputs_B);
    }

    ggml_opt_build(result);

    return result;
}

void ggml_opt_free(ggml_opt_context_t opt_ctx) {
    if (opt_ctx == nullptr) {
        return;
    }
    ggml_backend_buffer_free(opt_ctx->buf_static);
    ggml_backend_buffer_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    delete opt_ctx;
}

void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer) {
    if (optimizer) {
        ggml_graph_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } else {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

bool ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->static_graphs;
}

struct ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->inputs;
}

struct ggml_tensor * ggml_opt_outputs_A(ggml_opt_context_t opt_ctx) {
    return opt_ctx->outputs_A;
}

struct ggml_tensor * ggml_opt_outputs_B(ggml_opt_context_t opt_ctx) {
    return opt_ctx->outputs_B;
}

struct ggml_tensor * ggml_opt_labels_A(ggml_opt_context_t opt_ctx) {
    return opt_ctx->labels_A;
}

struct ggml_tensor * ggml_opt_labels_B(ggml_opt_context_t opt_ctx) {
    return opt_ctx->labels_B;
}

struct ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    return opt_ctx->loss;
}

struct ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    return opt_ctx->pred;
}

struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    return opt_ctx->ncorrect;
}

struct ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, struct ggml_tensor * node) {
    return ggml_graph_get_grad_acc(opt_ctx->gb_opt, node);
}

// ====== Optimization Result ======

ggml_opt_result_t ggml_opt_result_init() {
    return new ggml_opt_result;
}

void ggml_opt_result_free(ggml_opt_result_t result) {
    delete result;
}

void ggml_opt_result_reset(ggml_opt_result_t result) {
    result->ndata = 0;
    result->loss.clear();
    result->pred.clear();
    result->ncorrect = 0;
}

void ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata) {
    *ndata = result->ndata;
}

void ggml_opt_result_loss(ggml_opt_result_t result, double * loss, double * unc) {
    const int64_t nbatches = result->loss.size(); // Number of physical batches.

    if (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        return;
    }

    double sum         = 0.0;
    double sum_squared = 0.0;

    for (const float & loss : result->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        const float loss_scaled = result->loss_per_datapoint ? loss*result->opt_period : loss;
        sum         += loss_scaled;
        sum_squared += loss_scaled*loss_scaled;
    }

    const double mean = sum/nbatches;
    *loss = result->loss_per_datapoint ? mean : sum;

    if (!unc) {
        return;
    }

    if (nbatches < 2) {
        *unc = NAN;
        return;
    }

    const double var_sum = sum_squared/nbatches - mean*mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = result->loss_per_datapoint ? sqrt(var_sum / (nbatches - 1)) : sqrt(var_sum * nbatches/(nbatches - 1));
}

void ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred) {
    for (size_t i = 0; i < result->pred.size(); ++i) {
        pred[i] = result->pred[i];
    }
}

void ggml_opt_result_accuracy(ggml_opt_result_t result, double * accuracy, double * unc) {
    *accuracy = result->ncorrect >= 0 ? double(result->ncorrect) / double(result->ndata) : NAN;

    if (!unc) {
        return;
    }

    *unc = result->ncorrect >= 0 && result->ndata >= 2 ?
        sqrt((*accuracy) * (1.0 - (*accuracy)) / double(result->ndata - 1)) : NAN;
}

// ====== Computation ======

void ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        struct ggml_context * ctx_compute,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * inputs,
        struct ggml_tensor  * outputs_A,
        struct ggml_tensor  * outputs_B) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs_A   = outputs_A; // Renamed
    opt_ctx->outputs_B   = outputs_B; // Added
}

void ggml_opt_alloc(ggml_opt_context_t opt_ctx, bool backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
    if (backward) {
        const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_next == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } else {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    if (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    struct ggml_cgraph * graph = nullptr;
    switch (opt_ctx->build_type) {
        case GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } break;
        case GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } break;
        case GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } break;
    }
    GGML_ASSERT(graph);

    if (opt_ctx->allocated_graph == graph) {
        opt_ctx->eval_ready = true;
        return;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    if (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);
    } else {
        opt_ctx->allocated_graph_copy = graph;
    }

    ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;

    opt_ctx->eval_ready = true;
}

void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result) {
    GGML_ASSERT(opt_ctx->eval_ready);
    if (opt_ctx->allocated_graph == opt_ctx->gb_opt) {
        struct ggml_opt_optimizer_params opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        GGML_ASSERT(opt_pars.adamw.alpha >  0.0f);
        GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
        GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
        GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
        GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
        GGML_ASSERT(opt_pars.adamw.eps   >= 0.0f);
        GGML_ASSERT(opt_pars.adamw.wd    >= 0.0f);
        GGML_ASSERT(opt_pars.adamw.wd    <= 1.0f);

        // beta1, beta2 after applying warmup
        const float beta1h = 1.0f/(1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
        const float beta2h = 1.0f/(1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

        float * adamw_par_data = ggml_get_data_f32(opt_ctx->adamw_params);
        adamw_par_data[0] = opt_pars.adamw.alpha;
        adamw_par_data[1] = opt_pars.adamw.beta1;
        adamw_par_data[2] = opt_pars.adamw.beta2;
        adamw_par_data[3] = opt_pars.adamw.eps;
        adamw_par_data[4] = opt_pars.adamw.wd;
        adamw_par_data[5] = beta1h;
        adamw_par_data[6] = beta2h;
    }

    ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->iter += opt_ctx->allocated_graph == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    if (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = nullptr;
        opt_ctx->gb_grad              = nullptr;
        opt_ctx->gb_opt               = nullptr;
        opt_ctx->allocated_graph      = nullptr;
        opt_ctx->allocated_graph_copy = nullptr;
    }

    opt_ctx->eval_ready = false;

    if (!result) {
        return;
    }

    if (result->ndata == 0) {
        result->loss_per_datapoint = opt_ctx->loss_per_datapoint;
        result->opt_period         = opt_ctx->opt_period;
    } else {
        GGML_ASSERT(result->loss_per_datapoint == opt_ctx->loss_per_datapoint);
        GGML_ASSERT(result->opt_period         == opt_ctx->opt_period);
    }

    // ndata should be based on one of the outputs, e.g. outputs_A or outputs_B if it exists.
    struct ggml_tensor * ndata_ref_tensor = opt_ctx->outputs_A ? opt_ctx->outputs_A : opt_ctx->outputs_B;
    // If only loss_B is active, outputs_A might be null.
    if (!ndata_ref_tensor && opt_ctx->outputs_B) ndata_ref_tensor = opt_ctx->outputs_B;


    GGML_ASSERT(ndata_ref_tensor != nullptr && "At least one output tensor (outputs_A or outputs_B) must be present for ndata calculation in eval");
    const int64_t ndata_from_batch = ndata_ref_tensor->ne[1]; // Number of data points in the current batch

    // This assertion might be too strict if loss.size() is not reset when ndata is reset.
    // Let's assume result->ndata tracks total processed items, and loss.size() is number of batches so far for current result accumulation.
    // The logic should be: if result->ndata was 0 (start of new accumulation), then it's fine.
    // If result->ndata > 0, then previous batches must have had same ndata_from_batch size.
    if (result->ndata > 0 && !result->loss.empty()) { // Check if this is not the first batch for this result object
         GGML_ASSERT(result->ndata / (int64_t)result->loss.size() == ndata_from_batch && "varying batch size not supported");
    }
    result->ndata += ndata_from_batch;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    float loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    result->loss.push_back(loss);

    if (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata_from_batch); // Corrected variable name
        ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, ggml_nbytes(opt_ctx->pred));
        result->pred.insert(result->pred.end(), pred.begin(), pred.end());
    }

    if (!opt_ctx->ncorrect || result->ncorrect < 0) {
        result->ncorrect = -1;
        return;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    result->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

void ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_result_t       result_train,
        ggml_opt_result_t       result_eval,
        int64_t                 idata_split,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    struct ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    struct ggml_tensor * labels_A_batch_tensor = ggml_opt_labels_A(opt_ctx); // Use labels_A for the batch
    struct ggml_tensor * labels_B_batch_tensor = ggml_opt_labels_B(opt_ctx); // Use labels_B for the batch
    struct ggml_tensor * data_tensor_full   = ggml_opt_dataset_data(dataset); // Full dataset data tensor
    GGML_ASSERT(data_tensor_full->ne[0] == inputs->ne[0]); // Ensure feature size matches

    const int64_t ndata_total_items_in_dataset = ggml_opt_dataset_ndata(dataset); // Total items in dataset
    const int64_t items_per_batch_from_input_tensor = inputs->ne[1]; // Items per batch based on input tensor's shape

    GGML_ASSERT(ndata_total_items_in_dataset % items_per_batch_from_input_tensor == 0);
    const int64_t nbatches = ndata_total_items_in_dataset / items_per_batch_from_input_tensor;

    idata_split = idata_split < 0 ? ndata_total_items_in_dataset : idata_split;
    // Ensure idata_split aligns with batch boundaries, it's in terms of number of items from dataset
    GGML_ASSERT(idata_split % items_per_batch_from_input_tensor == 0);
    const int64_t ibatch_split = idata_split / items_per_batch_from_input_tensor;


    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    for (; ibatch < ibatch_split; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ true);
        ggml_opt_dataset_get_batch(dataset, inputs, labels_A_batch_tensor, labels_B_batch_tensor, ibatch);
        ggml_opt_eval(opt_ctx, result_train);
        if (callback_train) {
            callback_train(true, opt_ctx, dataset, result_train, ibatch + 1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    for (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ false);
        ggml_opt_dataset_get_batch(dataset, inputs, labels_A_batch_tensor, labels_B_batch_tensor, ibatch);
        ggml_opt_eval(opt_ctx, result_eval);
        if (callback_eval) {
            callback_eval(false, opt_ctx, dataset, result_eval, ibatch + 1 - ibatch_split, nbatches - ibatch_split, t_loop_start);
        }
    }
}

void ggml_opt_epoch_callback_progress_bar(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_length = 8;
    const int64_t ibatch8 = 8 * ibatch;
    for (int64_t j = 0; j < bar_length; ++j) {
        if        (ibatch_max * (8*j + 8) / bar_length < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } else if (ibatch_max * (8*j + 7) / bar_length < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } else if (ibatch_max * (8*j + 6) / bar_length < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } else if (ibatch_max * (8*j + 5) / bar_length < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } else if (ibatch_max * (8*j + 4) / bar_length < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } else if (ibatch_max * (8*j + 3) / bar_length < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } else if (ibatch_max * (8*j + 2) / bar_length < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } else if (ibatch_max * (8*j + 1) / bar_length < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } else {
            fprintf(stderr, " ");
        }
    }

    const int64_t batch_size = ggml_opt_inputs(opt_ctx)->ne[1];
    const int64_t idata      = ibatch*batch_size;
    const int64_t idata_max  = ibatch_max*batch_size;

    double loss;
    double loss_unc;
    ggml_opt_result_loss(result, &loss, &loss_unc);

    double accuracy;
    double accuracy_unc;
    ggml_opt_result_accuracy(result, &accuracy, &accuracy_unc);

    const int64_t t_ibatch_us = ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    const int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    const int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    const int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    const int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    const int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    if (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_UNUSED(dataset);
}

void ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        enum ggml_opt_loss_type         loss_type,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        float                           val_split,
        bool                            silent) {
    ggml_time_init();
    const int64_t t_start_us = ggml_time_us();

    const int64_t ndata_total_items_in_dataset = ggml_opt_dataset_ndata(dataset);
    const int64_t nbatch_physical_from_inputs = inputs->ne[1]; // items per physical batch based on inputs tensor
    GGML_ASSERT(ndata_total_items_in_dataset % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical_from_inputs == 0);

    const int64_t opt_period       = nbatch_logical / nbatch_physical_from_inputs;
    const int64_t nbatches_logical = ndata_total_items_in_dataset / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split <  1.0f);
    const int64_t ibatch_split_logical = int64_t(((1.0f - val_split) * nbatches_logical));
    // Renamed idata_split_items to idata_split_arg_for_epoch
    const int64_t idata_split_arg_for_epoch = ibatch_split_logical * nbatch_logical;

    int64_t epoch = 1;

    // Assuming loss_type is for loss_A and a default (e.g. CE) for loss_B, or this needs adjustment
    ggml_opt_params params = ggml_opt_default_params(backend_sched, loss_type, GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    params.ctx_compute     = ctx_compute;
    params.inputs          = inputs;
    params.outputs_A       = outputs; // Assuming 'outputs' passed to ggml_opt_fit is outputs_A
    // params.outputs_B needs to be set if a second loss is used, ggml_opt_fit might need extension
    params.opt_period      = opt_period;
    params.get_opt_pars    = get_opt_pars;
    params.get_opt_pars_ud = &epoch;
    // If only one loss type is passed to ggml_opt_fit, we might want to set loss_B_weight to 0
    // For now, assume loss_type corresponds to loss_A and loss_B is secondary (e.g. CE with weight 1.0 or 0.0)
    // If 'outputs' is null, then this whole setup is problematic.
    // If 'loss_type' is meant for 'outputs_B', then 'outputs_A' (for hidden state loss) needs to be plumbed in.
    // This function might be too high-level for dual loss without modification to its signature.
    // For this refactoring, we'll assume params are correctly set up by the caller of ggml_opt_fit
    // regarding outputs_A, outputs_B, loss_type_A, loss_type_B, and weights.
    // The current signature of ggml_opt_fit only takes one 'outputs' and one 'loss_type'.
    // This implies it's for a single-loss setup. The refactor of ggml_opt_context
    // allows dual loss, but high-level functions like ggml_opt_fit would need changes
    // to fully utilize it. The change below assumes 'outputs' is 'outputs_A'.

    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    // Initial shuffle of the entire dataset if it's larger than one logical batch.
    if (nbatch_logical < ndata_total_items_in_dataset) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1);
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_val   = ggml_opt_result_init();

    ggml_opt_epoch_callback epoch_callback = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;

    for (; epoch <= nepoch; ++epoch) {
        // If the training data part (which could be the whole dataset if no validation split)
        // is larger than one logical batch, then it should be shuffled.
        // The initial shuffle (if nbatch_logical < ndata_total_items_in_dataset) handles the first epoch's full shuffle.
        // For subsequent epochs, or if the training set is smaller than the full dataset but still multi-batch,
        // this shuffles the training portion.
        if (nbatch_logical < idata_split_arg_for_epoch) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split_arg_for_epoch);
        } else if (idata_split_arg_for_epoch == ndata_total_items_in_dataset && nbatch_logical < ndata_total_items_in_dataset && epoch > 1) {
            // If the training set is the entire dataset, and it's multi-batch,
            // re-shuffle the entire dataset for epochs after the first.
            // The initial shuffle already handled epoch 1.
            ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all
        }
        // Note: If idata_split_arg_for_epoch is 0 (all validation), no shuffling happens here, which is correct.

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);

        if (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        // Corrected call to use idata_split_arg_for_epoch
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val, idata_split_arg_for_epoch, epoch_callback, epoch_callback);
        if (!silent) {
            fprintf(stderr, "\n");
        }
    }

    if (!silent) {
        int64_t t_total_s = (ggml_time_us() - t_start_us) / 1000000;
        const int64_t t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        const int64_t t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_total_h, t_total_m, t_total_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);
}
