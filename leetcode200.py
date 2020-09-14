'''
101. Symmetric Tree
binaryTree中有收录，可用bfs，这里用了dfs
'''

def isSymmetric(self, root: TreeNode) -> bool:
    if not root:
        return True

    return self.helper(root.left, root.right)

def helper(self, root1, root2):
    if not root1 and not root2:
        return True
        
    if root1 and root2:
        if root1.val != root2.val:
            return False
        if self.helper(root1.left, root2.right) and self.helper(root1.right, root2.left):
            return True
    return False

'''
102. Binary Tree Level Order Traversal
bfs
'''
def levelOrder(self, root: TreeNode) -> List[List[int]]:
    if not root:
        return []
    res = []
    self.bfs(root, res)
    return res
    
def bfs(self, root, res):
    queue = collections.deque([root])
    
    while queue:
        n = len(queue)
        level = []
        while n:
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            n -= 1
        res.append(level)


def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
    if not root:
        return []
    res =[]
    self.bfs(root, res)
    return res

def bfs(self, root, res):
    queue = collections.deque([root])
    direction = 1
    while queue:
        n = len(queue)
        level= []
        while n:
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            n -= 1
        if direction == 1:
            res.append(level)
        else:
            res.append(level[::-1])
        direction = -direction

'''
方法二：
通过调整元素append进入的位置来调整，避免了使用level[::-1]（时间复杂度为O(n)）
'''

    def bfs(self, root, res):
        queue = collections.deque([root])
        direction = True
        while queue:
            n = len(queue)
            level= []
            while n:
                if direction:
                    node = queue.popleft()
                    if node.left:
                        queue.append(node.left)
                    if node.right:
                        queue.append(node.right)
                else:
                    node = queue.pop()
                    if node.right:
                        queue.appendleft(node.right)
                    if node.left:
                        queue.appendleft(node.left)
                level.append(node.val)
                n -= 1
            res.append(level)
            direction = not direction

'''
104. Maximum Depth of Binary Tree
简单题
'''
'''
107. Binary Tree Level Order Traversal II
简单题，用deque appendleft到res里即可
'''

'''
110. Balanced Binary Tree
'''
def isBalanced(self, root: TreeNode) -> bool:
    if not root:
        return True
    h1, h2, balance = self.helper(root)
    return balance

def helper(self, root):
    if not root:
        return -1, -1, True
    
    l1, r1, leftBalance = self.helper(root.left)
    l2, r2, rightBalance = self.helper(root.right)
    
    if leftBalance and rightBalance:
        leftHeight = max(l1, r1) + 1
        rightHeight = max(l2, r2) + 1
        return leftHeight, rightHeight, abs(leftHeight - rightHeight) <= 1
    
    return 0, 0, False

'''
111. Minimum Depth of Binary Tree
注意： [3, 2] return值是2
'''

def minDepth(self, root: TreeNode) -> int:
    if not root:
        return 0
    
    left = self.minDepth(root.left)
    right = self.minDepth(root.right)
    
    if left == 0 and right == 0:
        return 1
    if left and right:
        return min(left, right) + 1
    if left:
        return left + 1
    if right:
        return right + 1

'''
105. Construct Binary Tree from Preorder and Inorder Traversal
利用preorder的特性先找到root，然后在inorder中找到root，左边的数字都在左子树，
右边的数字都在右子树
'''
def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
    if not inorder or len(inorder) == 0:
        return None

    n = len(preorder)
    return self.helper(0, 0, n - 1, preorder, inorder)

def helper(self, preStart, inStart, inEnd, preorder, inorder):
    if preStart > len(preorder) - 1:
        return None
    if inStart > inEnd:
        return None
    
    root = TreeNode(preorder[preStart])
    
    index = inorder.index(root.val)
    
    leftNodes = index - inStart
    
    leftTree = self.helper(preStart + 1, inStart, index - 1, preorder, inorder)
    rightTree = self.helper(preStart + leftNodes + 1, index + 1, inEnd, preorder, inorder)
    
    root.left = leftTree
    root.right = rightTree
    
    return root

'''
106. Construct Binary Tree from Inorder and Postorder Traversal
与105类似解法
'''
def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
    if not postorder or not inorder or len(inorder) == 0 or len(postorder) == 0:
        return None
    return self.helper(inorder, postorder, 0, len(inorder) - 1, len(postorder) - 1)
    
def helper(self, inorder, postorder, inStart, inEnd, postStart):
    if inStart > inEnd:
        return None
    if postStart < 0:
        return None
    
    root = TreeNode(postorder[postStart])
    
    index = inorder.index(root.val)
    nodesRight = inEnd - index 
    
    rightTree = self.helper(inorder, postorder, index + 1, inEnd, postStart - 1)
    leftTree = self.helper(inorder, postorder, inStart, index - 1, postStart - nodesRight - 1)
    
    root.left = leftTree
    root.right = rightTree
    
    return root

'''
108. Convert Sorted Array to Binary Search Tree
简单题 找到中间点作为root再分别处理左右两边节点
'''
def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
    if not nums or len(nums) == 0:
        return None
    
    return self.helper(0, len(nums) - 1, nums)

def helper(self, start, end, nums):
    if start > end:
        return None
    
    mid = (start + end) // 2
    root = TreeNode(nums[mid])
    
    root.left = self.helper(start, mid - 1, nums)
    root.right = self.helper(mid + 1, end, nums)
    
    return root

'''
109. Convert Sorted List to Binary Search Tree
还是要找到中间节点作为root再处理两边节点
注意需要断开root左边的链接
错误点：没有注意到head是永远不变的，所以要判断是否已经把head作为当前root
如果是的话左子树当前因该返回None
time : O(nlogn) space: logn 
'''

def sortedListToBST(self, head: ListNode) -> TreeNode:
        if not head:
            return None
        return self.helper(head)
    
    def helper(self, head):
        if not head:
            return None
        
        mid = self.findRoot(head)
        root = TreeNode(mid.val)
        
        if mid == head:
            root.left = None
        else:
            root.left = self.helper(head)
        root.right = self.helper(mid.next)
        
        return root
    
    def findRoot(self, head):
        dummy = None
        slow = fast = head
        while fast and fast.next:
            dummy = slow
            slow = slow.next
            fast = fast.next.next
        if dummy:
            dummy.next = None
        return slow

'''
113. Path Sum II
因为并没有到底，所以res.append(curr)之后不能return，
否则会导致答案中有没有pop出的元素
需要二次过
'''
def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
    if not root:
        return []
    res = []
    self.helper(root, sum, [], res, 0)
    return res

def helper(self, root, sum, curr, res, total):
    if not root:
        return 
    
    curr.append(root.val)
    total += root.val
    
    if not root.left and not root.right and total == sum:
        res.append(curr[:])
    
    self.helper(root.left, sum, curr, res, total)
    self.helper(root.right, sum, curr, res, total)
    curr.pop()

'''
114. Flatten Binary Tree to Linked List
注意点：链接左右两边flatten过后的list时要注意断开和连接的操作 
需二刷
'''
def flatten(self, root: TreeNode) -> None:

    if not root:
        return None
    
    head = root
    self.flatten(root.left)
    self.flatten(root.right)
    
    if root.left:
        rightNode = root.right
        root.right = root.left
        root.left = None
        while root and root.right:
            root = root.right
        
        root.right = rightNode
    return head

def flatten(self, root: TreeNode) -> None:
    """
    Do not return anything, modify root in-place instead.
    """
    self.helper(root)

def helper(self, root):
    if not root:
        return None, None
    lhead, ltail = self.helper(root.left)
    rhead, rtail = self.helper(root.right)
    root.left = None
    if lhead and rhead:
        root.right = lhead
        ltail.right = rhead
        return root, rtail
    if lhead:
        root.right = lhead
        return root, ltail
    if rhead:
        return root, rtail
    return root, root

