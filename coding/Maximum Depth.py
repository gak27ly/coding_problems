# 求二叉树最大深度

def maxDepth(self, root):
    # write your code here
    if not root:
        return 0
        
    
    leftMax = self.maxDepth(root.left)
    rightMax = self.maxDepth(root.right)
    
    if leftMax > rightMax:
        return leftMax + 1
        
    else:
        return rightMax + 1


def maxDepth(self, root):
    if not root:
        return 0

    return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1