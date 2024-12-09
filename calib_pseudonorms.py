def estimatePseudonormalsCalibrated(I, L):

   

    B = None





    B = np.linalg.pinv(L.T)@I




    return B
