int ix_base = int(coord[2]) - pad;
int iy_base = int(coord[3]) - pad;
float sum = 0.f;
for (int ic = 0; ic < input_channels; ++ic)
for (int fy = 0; fy < filter_height; ++fy)
for (int fx = 0; fx < filter_width; ++fx) {
    float weight = input1[base1 + dot(strides1, ivec3(ic, fy, fx))];

    int sx = ix_base + fx;
    int sy = iy_base + fy;
    if (uint(sx) < uint(input_width) && uint(sy) < uint(input_height)) {
        float value = input0[base0 + dot(strides0, ivec3(ic, sy, sx))];
        sum += weight*value;
    }
}
output0[gl_GlobalInvocationID.x] = sum;
