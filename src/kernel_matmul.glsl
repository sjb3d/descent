// float load_a(uvec2 coord);
// float load_b(uvec2 coord);
// void store_c(uvec2 coord, float value);
// const uint K
// const uint TILE_M
// const uint TILE_N
// const uint TILE_K
// const uint GROUP_SIZE

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

const uint K_TILE_COUNT = (K + (TILE_K - 1))/TILE_K;

shared float s_a[A_TILE_H * A_TILE_STRIDE];
shared float s_b[B_TILE_H * B_TILE_STRIDE];

void main() {
    uvec2 c_tile_coord = gl_WorkGroupID.xy;
    uint thread_index = gl_LocalInvocationID.x;

    float result[C_VALUES_PER_THREAD];
    for (uint i = 0; i < C_VALUES_PER_THREAD; ++i) {
        result[i] = 0.f;
    }

    for (uint k_tile_index = 0; k_tile_index < K_TILE_COUNT; ++k_tile_index) {
        barrier();
        for (uint load_index = thread_index; load_index < A_TILE_SIZE; load_index += GROUP_SIZE) {
            uvec2 a_coord_in_tile = uvec2(load_index % A_TILE_W, load_index/A_TILE_W);
            uvec2 a_tile_coord = uvec2(k_tile_index, c_tile_coord.y);
            float a = load_a(a_tile_coord*uvec2(A_TILE_W, A_TILE_H) + a_coord_in_tile);
            s_a[a_coord_in_tile.y*A_TILE_STRIDE + a_coord_in_tile.x] = a;
        }
        for (uint load_index = thread_index; load_index < B_TILE_SIZE; load_index += GROUP_SIZE) {
            uvec2 b_coord_in_tile = uvec2(load_index % B_TILE_W, load_index/B_TILE_W);
            uvec2 b_tile_coord = uvec2(c_tile_coord.x, k_tile_index);
            float b = load_b(b_tile_coord*uvec2(B_TILE_W, B_TILE_H) + b_coord_in_tile);
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
        store_c(c_tile_coord*uvec2(C_TILE_W, C_TILE_H) + c_coord_in_tile, result[i]);
    }   
}
