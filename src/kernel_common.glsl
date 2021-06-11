#version 460 core

bool compute_grid_coord(out uint coord[1], uint shape0)
{
    uint tmp0 = gl_GlobalInvocationID.x;
    if (tmp0 >= shape0) {
        return false;
    }

    coord[0] = tmp0;
    return true;
}

bool compute_grid_coord(out uint coord[2], uint shape0, uint shape1)
{
    uint tmp1 = gl_GlobalInvocationID.x;
    if (tmp1 >= shape0*shape1) {
        return false;
    }

    uint tmp0 = tmp1/shape1;

    coord[0] = tmp0;
    coord[1] = tmp1 - shape1*tmp0;
}
