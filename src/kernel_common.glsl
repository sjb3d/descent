#version 460 core

void compute_grid_coord(out int coord[1], uint /*shape0*/)
{
    coord[0] = int(gl_GlobalInvocationID.x);
}

void compute_grid_coord(out int coord[2], uint /*shape0*/, uint shape1)
{
    uint remain = gl_GlobalInvocationID.x;

    uint tmp1 = remain;
    remain /= shape1;
    tmp1 -= remain*shape1;

    uint tmp0 = remain;

    coord[0] = int(tmp0);
    coord[1] = int(tmp1);
}

void compute_grid_coord(out int coord[4], uint /*shape0*/, uint shape1, uint shape2, uint shape3)
{
    uint remain = gl_GlobalInvocationID.x;

    uint tmp3 = remain;
    remain /= shape3;
    tmp3 -= remain*shape3;

    uint tmp2 = remain;
    remain /= shape2;
    tmp2 -= remain*shape2;

    uint tmp1 = remain;
    remain /= shape1;
    tmp1 -= remain*shape1;

    uint tmp0 = remain;

    coord[0] = int(tmp0);
    coord[1] = int(tmp1);
    coord[2] = int(tmp2);
    coord[3] = int(tmp3);
}

int dot(ivec3 a, ivec3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
