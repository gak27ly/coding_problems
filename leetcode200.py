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