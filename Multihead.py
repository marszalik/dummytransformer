import numpy as np

class Multihead:

  def __init__(self):
    pass

  def combineHeads(self, arrayOfHeadOutputs):
    result = np.array(arrayOfHeadOutputs[0])
    #make sure that the number of dimensions for this them variable is the same as headoutputs we want to add by taking the first element
    for headOutput in arrayOfHeadOutputs[1:]:
      #we skipped first element because we have it in result already
      result = np.concatenate((result, headOutput), -1)
    #second argument tells how to concatenate the arrays (by which axis)
    return result
