from Multihead import Multihead
# Test 1
multi = Multihead()
arrayOfHeads = [[[1, 2], [3, 3]], [[5, 6], [7, 7]], [[8, 9], [0, 0]]]
head1 = [[1, 2], [3, 3]]
head2 = [[5, 6], [7, 7]]
head3 = [[8, 9], [0, 0]]
multi.add_outupt_from_head(head1)
multi.add_outupt_from_head(head2)
multi.add_outupt_from_head(head3)

print(multi.combineHeads())