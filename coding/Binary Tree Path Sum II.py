# 246. Binary Tree Path Sum II

# Your are given a binary tree in which each node contains a value. 
# Design an algorithm to get all paths which sum to a given value. 
# The path does not need to start or end at the root or a leaf, 
# but it must go in a straight line down.

def binaryTreePathSum2(self, root, target):
    # Write your code here
    result = []
    path = []
    if root is None:
        return result
    self.dfs(root, path, result, 0,  target)
    return result

#用l来记录当前path的长度， 然后没加入一个数字就看是否能够从这个数字向上构成一个含有target的subpath
def dfs(self, root, path, result, l, target):
    if root is None:
        return
    path.append(root.val)
    tmp = target
    for i in xrange(l , -1, -1):
        tmp -= path[i]
        if tmp == 0:
            result.append(path[i:])

    self.dfs(root.left, path, result, l + 1, target)
    self.dfs(root.right, path, result, l + 1, target)

    path.pop()