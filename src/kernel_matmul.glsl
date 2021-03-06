// float load_a(uint batch_index, uvec2 coord);
// float load_b(uint batch_index, uvec2 coord);
// void store_c(uint batch_index, uint k_chunk_index, uvec2 coord, float value);
// const uint M
// const uint N
// const uint K
// const uint TILE_M
// const uint TILE_N
// const uint TILE_K
// const uint GROUP_SIZE
// const uint K_CHUNK_SIZE_IN_TILES
// const uint K_CHUNK_COUNT
// const uint BATCH_COUNT
// const bool LOAD_A_IN_COLUMNS
// const bool LOAD_B_IN_COLUMNS

layout(local_size_x = GROUP_SIZE) in;

const uint A_TILE_W = TILE_K;
const uint A_TILE_H = TILE_M;
const uint A_TILE_SIZE = A_TILE_W * A_TILE_H;

const uint B_TILE_W = TILE_N;
const uint B_TILE_H = TILE_K;
const uint B_TILE_SIZE = B_TILE_W * B_TILE_H;

const uint SHARED_PAD = 1;
const uint A_TILE_STRIDE = A_TILE_W + SHARED_PAD;
const uint B_TILE_STRIDE = B_TILE_W + SHARED_PAD;

const uint C_TILE_W = TILE_N;
const uint C_TILE_H = TILE_M;
const uint C_TILE_SIZE = C_TILE_W * C_TILE_H;
const uint C_VALUES_PER_THREAD = C_TILE_SIZE / GROUP_SIZE; // must divide exactly!

const uint M_TILE_COUNT = (M + (TILE_M - 1))/TILE_M;
const uint N_TILE_COUNT = (N + (TILE_N - 1))/TILE_N;
const uint K_TILE_COUNT = (K + (TILE_K - 1))/TILE_K;

shared float s_a[A_TILE_H * A_TILE_STRIDE];
shared float s_b[B_TILE_H * B_TILE_STRIDE];

void main() {
    int c_output_coord[4];
    compute_grid_coord(gl_WorkGroupID.x, c_output_coord, K_CHUNK_COUNT, BATCH_COUNT, M_TILE_COUNT, N_TILE_COUNT);
    uvec2 c_tile_coord = uvec2(c_output_coord[3], c_output_coord[2]);
    uint batch_index = c_output_coord[1];
    uint k_chunk_index = c_output_coord[0];

    uint thread_index = gl_LocalInvocationID.x;

    float result[C_VALUES_PER_THREAD];
    for (uint i = 0; i < C_VALUES_PER_THREAD; ++i) {
        result[i] = 0.f;
    }

    uint k_tile_begin = k_chunk_index * K_CHUNK_SIZE_IN_TILES;
    uint k_tile_end = min(k_tile_begin + K_CHUNK_SIZE_IN_TILES, K_TILE_COUNT);
    for (uint k_tile_index = k_tile_begin; k_tile_index != k_tile_end; ++k_tile_index) {
        barrier();
        for (uint load_index = thread_index; load_index < A_TILE_SIZE; load_index += GROUP_SIZE) {
            uvec2 a_coord_in_tile = LOAD_A_IN_COLUMNS
                ? uvec2(load_index/A_TILE_H, load_index % A_TILE_H)
                : uvec2(load_index % A_TILE_W, load_index/A_TILE_W);
            uvec2 a_tile_coord = uvec2(k_tile_index, c_tile_coord.y);
            float a = load_a(batch_index, a_tile_coord*uvec2(A_TILE_W, A_TILE_H) + a_coord_in_tile);
            s_a[a_coord_in_tile.y*A_TILE_STRIDE + a_coord_in_tile.x] = a;
        }
        for (uint load_index = thread_index; load_index < B_TILE_SIZE; load_index += GROUP_SIZE) {
            uvec2 b_coord_in_tile = LOAD_B_IN_COLUMNS
                ? uvec2(load_index/B_TILE_H, load_index % B_TILE_H)
                : uvec2(load_index % B_TILE_W, load_index/B_TILE_W);
            uvec2 b_tile_coord = uvec2(c_tile_coord.x, k_tile_index);
            float b = load_b(batch_index, b_tile_coord*uvec2(B_TILE_W, B_TILE_H) + b_coord_in_tile);
            s_b[b_coord_in_tile.y*B_TILE_STRIDE + b_coord_in_tile.x] = b;
        }
        barrier();

        for (uint k_index = 0; k_index < TILE_K; ++k_index) {
            for (uint i = 0; i < C_VALUES_PER_THREAD; ++i) {
                uint c_index = i*GROUP_SIZE + thread_index;
                uvec2 c_coord_in_tile = uvec2(c_index % C_TILE_W, c_index / C_TILE_W);
                uvec2 a_coord_in_tile = uvec2(k_index, c_coord_in_tile.y);
                uvec2 b_coord_in_tile = uvec2(c_coord_in_tile.x, k_index);
                float a = s_a[a_coord_in_tile.y*A_TILE_STRIDE + a_coord_in_tile.x];
                float b = s_b[b_coord_in_tile.y*B_TILE_STRIDE + b_coord_in_tile.x];
                result[i] += a*b;
            }
        }
    }

    for (uint i = 0; i < C_VALUES_PER_THREAD; ++i) {
        uint c_index = i*GROUP_SIZE + thread_index;
        uvec2 c_coord_in_tile = uvec2(c_index % C_TILE_W, c_index / C_TILE_W);
        store_c(k_chunk_index, batch_index, c_tile_coord*uvec2(C_TILE_W, C_TILE_H) + c_coord_in_tile, result[i]);
    }   
}
