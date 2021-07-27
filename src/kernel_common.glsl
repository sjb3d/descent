#version 460 core

void compute_grid_coord(
    uint remain,
    out int coord[1],
    uint /*shape0*/)
{
    coord[0] = int(remain);
}

void compute_grid_coord(
    uint remain,
    out int coord[2],
    uint /*shape0*/,
    uint shape1)
{
    uint tmp1 = remain;
    remain /= shape1;
    tmp1 -= remain*shape1;

    uint tmp0 = remain;

    coord[0] = int(tmp0);
    coord[1] = int(tmp1);
}

void compute_grid_coord(
    uint remain,
    out int coord[3],
    uint /*shape0*/,
    uint shape1,
    uint shape2)
{
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
}

void compute_grid_coord(
    uint remain,
    out int coord[4],
    uint /*shape0*/,
    uint shape1,
    uint shape2,
    uint shape3)
{
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

void compute_grid_coord(
    uint remain,
    out int coord[5],
    uint /*shape0*/,
    uint shape1,
    uint shape2,
    uint shape3,
    uint shape4)
{
    uint tmp4 = remain;
    remain /= shape4;
    tmp4 -= remain*shape4;

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
    coord[4] = int(tmp4);
}

void compute_grid_coord(
    uint remain,
    out int coord[6],
    uint /*shape0*/,
    uint shape1,
    uint shape2,
    uint shape3,
    uint shape4,
    uint shape5)
{
    uint tmp5 = remain;
    remain /= shape5;
    tmp5 -= remain*shape5;

    uint tmp4 = remain;
    remain /= shape4;
    tmp4 -= remain*shape4;

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
    coord[4] = int(tmp4);
    coord[5] = int(tmp5);
}

void compute_grid_coord(
    uint remain,
    out int coord[7],
    uint /*shape0*/,
    uint shape1,
    uint shape2,
    uint shape3,
    uint shape4,
    uint shape5,
    uint shape6)
{
    uint tmp6 = remain;
    remain /= shape6;
    tmp6 -= remain*shape6;

    uint tmp5 = remain;
    remain /= shape5;
    tmp5 -= remain*shape5;

    uint tmp4 = remain;
    remain /= shape4;
    tmp4 -= remain*shape4;

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
    coord[4] = int(tmp4);
    coord[5] = int(tmp5);
    coord[6] = int(tmp6);
}

int dot(ivec3 a, ivec3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

layout(push_constant) uniform constants
{
	uint rand_seed;
};

uint pcg(uint v)
{
    uint state = v*747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state)*277803737u;
    return (word >> 22u) ^ word;
}

float rand_from_index(uint uid, int index)
{
    uint hash = pcg(pcg(index) + rand_seed + uid);
    return float(hash)/float(0xffffffffu);
}
