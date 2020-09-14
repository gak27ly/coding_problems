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