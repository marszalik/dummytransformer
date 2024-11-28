import numpy as np


class Multihead:

  def __init__(self):
    self.arrayOfHeadOutputs = []
    

  def combineHeads(self):
    if self.arrayOfHeadOutputs == []:
      print("No heads to add, please add heads first")
      return
    
    result = np.array(self.arrayOfHeadOutputs[0])
    #make sure that the number of dimensions for this them variable is the same as headoutputs we want to add by taking the first element
    for headOutput in self.arrayOfHeadOutputs[1:]:
      #we skipped first element because we have it in result already
      result = np.concatenate((result, headOutput), -1)
    #second argument tells how to concatenate the arrays (by which axis)
    return result

  def add_outupt_from_head(self, headOutput):
    if self.arrayOfHeadOutputs == []:
      self.arrayOfHeadOutputs = [headOutput]
    else:
      self.arrayOfHeadOutputs.append(headOutput)
