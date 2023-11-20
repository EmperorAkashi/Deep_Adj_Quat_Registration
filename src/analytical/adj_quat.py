import numpy as np

def make_dets(X:np.ndarray, U:np.ndarray) -> np.ndarray:
    """calculate profile matrix
    so that 10 determinants which encodes the rotation mat
    X: source point cloud
    U: target point cloud
    """
    XX = X.T[0].dot(X.T[0])
    XY = X.T[0].dot(X.T[1])
    XZ = X.T[0].dot(X.T[2])

    YY = X.T[1].dot(X.T[1])
    YZ = X.T[1].dot(X.T[2])
    ZZ = X.T[2].dot(X.T[2])

    UX = U.T[0].dot(X.T[0])
    UY = U.T[0].dot(X.T[1])
    UZ = U.T[0].dot(X.T[2])
    
    VX = U.T[1].dot(X.T[0])
    VY = U.T[1].dot(X.T[1])
    VZ = U.T[1].dot(X.T[2])
    
    d1 = np.linalg.det(np.array([[XX,XY,XZ],[XY,YY,YZ],[XZ,YZ,ZZ]])) 
    d2 = np.linalg.det(np.array([[XX,XY,UX],[XY,YY,UY],[XZ,YZ,UZ]]))
    d3 = np.linalg.det(np.array([[XX,XY,VX],[XY,YY,VY],[XZ,YZ,VZ]]))

    d4 = np.linalg.det(np.array([[XX,XZ,UX],[XY,YZ,UY],[XZ,ZZ,UZ]]))
    d5 = np.linalg.det(np.array([[XX,XZ,VX],[XY,YZ,VY],[XZ,ZZ,VZ]]))
    d6 = np.linalg.det(np.array([[XX,UX,VX],[XY,UY,VY],[XZ,UZ,VZ]]))
    
    d7 = np.linalg.det(np.array([[XY,XZ,UX],[YY,YZ,UY],[YZ,ZZ,UZ]]))
    d8 = np.linalg.det(np.array([[XY,XZ,VX],[YY,YZ,VY],[YZ,ZZ,VZ]]))
    d9 = np.linalg.det(np.array([[XY,UX,VX],[YY,UY,VY],[YZ,UZ,VZ]]))
    d10 = np.linalg.det(np.array([[XZ,UX,VX],[YZ,UY,VY],[ZZ,UZ,VZ]]))

    return d1, d2, d3, d4, d5, d6, d7, d8, d9, d10

def make_R_tilde(X:np.ndarray, U:np.ndarray) -> np.ndarray:
    """obtain rotation matrix from 10 determinants
    """
    d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 = make_dets(X,U)

    R_tilde = np.array([[d7/d1, -d4/d1, d2/d1],
                        [d8/d1, -d5/d1, d3/d1],
                        [d6/d1,  d9/d1, d10/d1]])
    
    return R_tilde
