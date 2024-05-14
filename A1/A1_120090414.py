import numpy as np


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_120090414(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :w type: numpy.ndarray
    :XT type: numpy.ndarray
    :InvXTX type: numpy.ndarray
   
    """
    # your code goes here
    XT = X.T
    XTX = np.dot(XT,X)
    InvXTX = np.linalg.inv(XTX)
    XTy = np.dot(XT,y)
    w = np.dot(InvXTX, XTy)

    # return in this order
    return w, XT, InvXTX
