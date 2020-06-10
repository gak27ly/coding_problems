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
方法1：设置全局变量，用helper function遍历整个二叉树 求得每个node的sum
并不断地update最小值和最小node
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
'''
902. BST中第K小的元素
给一棵二叉搜索树，写一个 KthSmallest 函数来找到其中第 K 小的元素。

解法： 使用stack来存nodes， 一路向左将点加入栈中， 弹出一个点，看是否有右点，再将此右点的左点全部压入栈中。
依此顺序弹出k-1个点后栈顶的点就是第k大的元素。
'''
def kthSmallest(self, root, k):
    # write your code here
    if not root:
        return None
    node = root
    stack = []
    
    while node:
        stack.append(node)
        node = node.left

    for i in range(k-1):
        node = stack.pop()
        if node.right:
            node = node.right
            while node:
                stack.append(node)
                node = node.left
    
    return stack[-1].val

'''
578. Lowest Common Ancestor III

Given the root and two nodes in a Binary Tree. Find the lowest common ancestor(LCA) of the two nodes.
The lowest common ancestor is the node with largest depth which is the ancestor of both nodes.
Return null if LCA does not exist.
解法： 考虑返回的情况时，是否找到了node A and node B. 如果一个点找到了node A, node B， 就反回这个点作为lca.
'''
def lowestCommonAncestor3(self, root, A, B):
    # write your code here
    if not root:
        return None
    hasA, hasB, LCA = self.helper(root, A, B)
    return LCA

def helper(self, root, A, B):
    if not root:
        return False, False, None

    leftHasA, leftHasB, leftLca = self.helper(root.left, A, B)
    rightHasA, rightHasB, rightLca = self.helper(root.right, A, B)
    
    hasA = leftHasA or rightHasA or root == A
    hasB = leftHasB or rightHasB or root == B
    
    if leftLca:
        return hasA, hasB, leftLca
    
    if rightLca:
        return hasA, hasB, rightLca
    
    if hasA and hasB:
        return hasA, hasB, root

    return hasA, hasB, None



'''
95. Validate Binary Search Tree
Given a binary tree, determine if it is a valid binary search tree (BST).
Assume a BST is defined as follows:
The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
解法1: 设置上下边界，通过不断更改上下边界来判断当前node是否符合BST
'''
def isValidBST(self, root):
    # write your code here
    isValid = self.helper(root, sys.maxsize, -sys.maxsize + 1)
    return isValid
    
def helper(self, root, maxNum, minNum):
    if not root:
        return True
    
    if root.val >= maxNum or root.val <= minNum:
        return False
    
    return self.helper(root.left, root.val, minNum) and self.helper(root.right, maxNum, root.val)

'''
解法2: 中序遍历， 将所有左边node加入stack，记录一个last_node， 如果pop出来的parent小于last_node，return False
还需要看是否有right node，如果有也要加入下一轮循环
如果所有点都符合，最后return True
'''

def isValidBST(self, root):
    stack = []
    curr = root
    last_val = -sys.maxsize - 1
    
    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left
            
        curr = stack.pop()
        if curr.val <= last_val:
            return False

        last_val = curr.val
        curr = curr.right 
    
    return True

'''
Given a binary tree, find the subtree with maximum average. Return the root of the subtree.
解法： 分治法，得到子树的合，nodes数量，子树中的最大平均值和最大子树。
'''
def findSubtree2(self, root):
    # write your code here
    
    nodeSum, numOfNodes, maxAverge, maxNode = self.helper(root)
    
    return maxNode
    

def helper(self, root):
    if not root:
        return 0, 0, -sys.maxsize - 1, None
        
    leftSum, leftNodes, leftMaxAve, leftNode = self.helper(root.left)
    rightSum, rightNodes, rightMaxAve, rightNode = self.helper(root.right)
    
    nodeSum = root.val + leftSum + rightSum
    numOfNodes = leftNodes + rightNodes + 1
    average = nodeSum / numOfNodes
    
    if average == max(leftMaxAve, rightMaxAve, average):
        return nodeSum, numOfNodes, average, root
    
    if leftMaxAve == max(leftMaxAve, rightMaxAve, average):
        return nodeSum, numOfNodes, leftMaxAve, leftNode
    
    return nodeSum, numOfNodes, rightMaxAve, rightNode




# Binary Tree Path Sum 
# Your are given a binary tree in which each node contains a value. 
# Design an algorithm to get all paths which sum to a given value. 
# The path does not need to start or end at the root or a leaf,
#  but it must go in a straight line down.

def binaryTreePathSum2(self, root, target):
    # write your code here
    combination, res = [], []
    self.helper(root, target, 0, combination, res)

    return res
        

def helper(self, root, target, total, combination, res):
    if not root:
        return 
    
    total += root.val
    
    combination.append(root.val)
# //只有当到leaf的时候才能加入ressult
    if not root.left and not root.right and total == target:
        res.append(combination[:])

    self.helper(root.left, target, total, combination, res)
    self.helper(root.right, target, total, combination, res)

    combination.pop()

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
'''
11. Search Range in Binary Search Tree
Given a binary search tree and a range [k1, k2], return node values within a given range in ascending order.
解法： 利用inorder traerse 特点来记录符合要求的点
'''

def searchRange(self, root, k1, k2):
    # write your code here
    result = []
    self.traverse(root, k1, k2,result)
    return result
    
def traverse(self, root, k1, k2, result):
    
    if not root:
        return
    
    if root.val > k1:
        self.traverse(root.left, k1, k2, result)
    
    if root.val >= k1 and root.val <= k2:
        result.append(root.val)
    
    
    if root.val < k2:
            self.traverse(root.right, k1, k2, result)
