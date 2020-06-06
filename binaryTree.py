'''
1. 
Given a non-empty binary search tree and a target value, 
find the value in the BST that is closest to the target.

思路： 设置上下边界，在BST中不断向目标靠近， 最终得到上下边界值
'''
def closestValue(self, root, target):
    # write your code here
    high, low = root.val

    while root:
    	if root.val == target:
    		return root.val

    	if root.val > target:
    		high = root.val
    		root = root.left
    	else:
    		low = root.val
    		root = root.right
    if abs(high - target) > abs(low - target):
    	return low
    return high


'''
Given a binary tree, find the subtree with minimum sum. Return the root of the subtree.

Example:
Input:
{1,-5,2,1,2,-4,-5}
Output:1
Explanation:
The tree is look like this:
     1
   /   \
 -5     2
 / \   /  \
1   2 -4  -5 
The sum of whole tree is minimum, so return the root.
方法1：设置全局变量，用helper function遍历整个二叉树 不断地update最小值和最小node
'''
def findSubtree(self, root):
    # write your code here
    self.minSum = sys.maxsize
    self.minNode = None
    self.helper(root)
    
    return self.minNode  

def helper(self, root):
    if not root:
        return 0
    
    leftSum = self.helper(root.left)
    rightSum = self.helper(root.right)
    
    nodeSum = leftSum + rightSum + root.val
    
    if nodeSum < self.minSum:
        self.minSum = nodeSum
        self.minNode = root
        
    return nodeSum
''''
方法2：
纯分治法
思考需要什么返回值。 因为结果需要最小子树，随意要返回最小子树的和，此子树的root.但是因为这个子树不一定是最小的那个，可能上面还有更小的
所以要把当前root.val + leftsum + rightsum的值传上去.
'''
def findSubtree(self, root):
    # write your code here
    if not root:
        return None
        
    minSum, minNode, nodeSum = self.helper(root)
    
    return minNode
    

def helper(self, root):
    if not root:
        return sys.maxsize, None, 0
    
    leftMin, leftNode, leftSum = self.helper(root.left)
    rightMin, rightNode, rightSum = self.helper(root.right)
    
    nodeSum = leftSum + rightSum + root.val
    
    if leftMin == min(leftMin, rightMin, nodeSum):
        return leftMin, leftNode, nodeSum
    if rightMin == min(leftMin, rightMin, nodeSum):
        return rightMin, rightNode, nodeSum
    
    return nodeSum, root, nodeSum
'''
480. Binary Tree Paths
中文English
Given a binary tree, return all root-to-leaf paths.
Input：{1,2,3,#,5}
Output：["1->2->5","1->3"]
方法1： 分治法得到左子树path和右子树path
分别把str(root.val) + '->' + (path in leftPath + rightPath)加入path中
注意当在最后一个node时候会得到两个空的path， 此时需要返回str(root.val)
'''
def binaryTreePaths(self, root):
# write your code here
	if not root:
	    return []
	    
	leftPath = self.binaryTreePaths(root.left)
	rightPath = self.binaryTreePaths(root.right)

	if not leftPath and not rightPath:
	    return [str(root.val)]

	paths = []
	for path in leftPath:
	    paths.append(str(root.val) + '->' + path)
	    
	for path in rightPath:
	    paths.append(str(root.val) + '->' + path)
	    
	return path

'''
453 Flatten a binary tree to a fake "linked list" in pre-order traversal.

Here we use the right pointer in TreeNode as the next pointer in ListNode.
Input:{1,2,5,3,4,#,6}
Output：{1,#,2,#,3,#,4,#,5,#,6}
方法1： 利用stack 来实现inorder traverse
root -> right -> left
pop() 出来必然是先左后右
'''
def flatten(self, root):
    # write your code here
    if not root:
    	return None

    stack = collections.deque([root])

    while stack:
    	node = stack.pop()
    	if node.right:
    		stack.append(node.right)
    	if node.left:
    		stack.append(node.left)

    	node.left = None

    	if stack:
    		node.right = stack[-1]
    	else:
    		node.right = None

    return root

'''
93. Balanced Binary Tree

Given a binary tree, determine if it is height-balanced.
解法：确定是否是balance，就要看左右子树高度是否相差大于1.
需要return 左子树和右子树高度，还有这两个子树是否balance
'''
class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """
    def isBalanced(self, root):
        # write your code here
        
        height, balanced = self.helper(root)
        
        return balanced
    
    def helper(self, root):
        
        if not root:
            return 0, True
            
            
        leftHeight, leftBalance = self.helper(root.left)
        rightHeight, rightBalance = self.helper(root.right)
        
        if leftBalance  == False or rightBalance == False:
            return 0, False
        

        if abs(leftHeight - rightHeight) > 1:
            return max(leftHeight, rightHeight) + 1, False
        
        return max(leftHeight, rightHeight) + 1, True