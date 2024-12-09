def renderNDotLSphere(center, rad, light, pxSize, res):



    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl


    [X, Y] = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    X = (X - res[0]/2) * pxSize*1.e-4
    Y = (Y - res[1]/2) * pxSize*1.e-4
    Z = np.sqrt(rad**2+0j-X**2-Y**2)
    X[np.real(Z) == 0] = 0
    Y[np.real(Z) == 0] = 0
    Z = np.real(Z)

    image = None
 
    valid_mask = np.real(Z) > 0
    X[~valid_mask] = 0
    Y[~valid_mask] = 0
    Z[~valid_mask] = 0

    intensity_vals = np.zeros(X.shape)
    hemisphere_points = np.stack([X, Y, Z], axis = -1)
    for i in range(hemisphere_points.shape[0]):
        for j in range(hemisphere_points.shape[1]):
            point = hemisphere_points[i,j]


            s_normal = point - center
            s_normal = s_normal / np.linalg.norm(s_normal)
            intensity_vals[i,j] = max(0,np.dot(s_normal, light/np.linalg.norm(light)))
    image = intensity_vals

    return image

