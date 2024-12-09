def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = None
    normals = None

    ### YOUR CODE HERE
    albedos = np.zeros(B.shape[1])
    normals = np.zeros(B.shape)

    for p in range(B.shape[1]):
      b_p = B[:,p]
      albedos[p] = np.linalg.norm(b_p)
      if albedos[p] > 0:
        normals[:,p] = b_p/albedos[p]


    ### END YOUR CODE

    return albedos, normals
