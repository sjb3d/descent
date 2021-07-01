#version 460 core

void compute_grid_coord(out uint coord[1], uint /*shape0*/)
{
    coord[0] = gl_GlobalInvocationID.x;
}

void compute_grid_coord(out uint coord[2], uint /*shape0*/, uint shape1)
{
    uint remain = gl_GlobalInvocationID.x;

    uint tmp1 = remain;
    remain /= shape1;
    tmp1 -= remain*shape1;

    uint tmp0 = remain;

    coord[0] = tmp0;
    coord[1] = tmp1;
}
