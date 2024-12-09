def estimateShape(normals, s):

    surface = None



    Nx = normals[0, :].reshape(s)
    Ny = normals[1, :].reshape(s)
    Nz = normals[2, :].reshape(s)

    epsilon = 1e-6
    Nx[Nx == 0] = epsilon

    z_grad = - Nx/Nz
    y_grad = - Ny/Nz

    surface = integrateFrankot(z_grad, y_grad)

    return surface
