from Multihead import Multihead
# Test 1
multi = Multihead()
arrayOfHeads = [[[1, 2], [3, 3]], [[5, 6], [7, 7]], [[8, 9], [0, 0]]]
print(multi.combineHeads(arrayOfHeads))
