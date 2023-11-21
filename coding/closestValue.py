找BST中最接近target的值

设置上限和下限 根据bst特点找到最接近的两个点
def closestValue(self, root, target):
    # write your code here
    low, up = root.val, root.val
    
    while root:
    
        if target > root.val:
            low = root.val
            root = root.right
        else:
            up = root.val
            root = root.left
        
    if abs(up - target) < abs(target - low):
        return up
    return low