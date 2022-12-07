# Python3 program for Iterative Preorder
# Traversal of N-ary Tree.
# Preorder: Root, print children
# from left to right.

from collections import deque


# Node Structure of K-ary Tree
class NewNode():

    def __init__(self, val):
        self.key = val
        # all children are stored in a list
        self.child = {}

root = NewNode(1)
root.child[2] = NewNode(2)
root.child[3] = NewNode(3)
root.child[4] = NewNode(4)
root.child[2].child[5] = NewNode(5)
root.child[2].child[5].child[10] = NewNode(10)
#root.child[0].child.append(NewNode(6))
# root.child[0].child[1].child.append(NewNode(11))
# root.child[0].child[1].child.append(NewNode(12))
# root.child[0].child[1].child.append(NewNode(13))
root.child[4].child[7] = NewNode(7)
root.child[4].child[8] = NewNode(8)
#root.child[4].child.append(NewNode(9))

a = 1
#preorderTraversal(root)