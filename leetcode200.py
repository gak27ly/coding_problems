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
错误点：要用popleft()
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

'''
103. Binary Tree Zigzag Level Order Traversal
错误点： direction要设置在while loop外面
'''

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
107. Binary Tree Level Order Traversal II
简单题，用deque appendleft到res里即可
'''

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
错误点： 没有考虑到左节点为空或右节点为空的情况
需再刷
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

'''
116. Populating Next Right Pointers in Each Node
利用preorder，先连接左右节点，再在有next node的情况下连接中间节点
'''
def connect(self, root: 'Node') -> 'Node':
    if not root or not root.left:
        return root
    
    if root.left and root.right:
        root.left.next = root.right
    if root.next:
        root.right.next = root.next.left
    
    self.connect(root.left)
    self.connect(root.right)
    
    return root

'''
117. Populating Next Right Pointers in Each Node II
解法： 用firstNode表示每一行第一个node，currentNode代表当前node
根据条件利用helper function找到下一个连接的位置
'''

def connect(self, root: 'Node') -> 'Node':
    if not root:
        return root
    
    firstNode = root
    
    while firstNode:
        currentNode = firstNode
        while currentNode:
            if currentNode.left:
                if currentNode.right:
                    currentNode.left.next = currentNode.right
                else:
                    currentNode.left.next = self.findNext(currentNode)
            if currentNode.right:
                currentNode.right.next = self.findNext(currentNode)

            currentNode = currentNode.next
        firstNode = self.findNextLevel(firstNode)
    
    return root
        
    
def findNext(self, currentNode):
    if not currentNode:
        return None
    currentNode = currentNode.next
    while currentNode:
        if currentNode.left:
            return currentNode.left
        if currentNode.right:
            return currentNode.right
        currentNode = currentNode.next
    return currentNode

def findNextLevel(self, firstNode):
    while firstNode:
        if firstNode.left:
            return firstNode.left
        if firstNode.right:
            return firstNode.right
        firstNode = firstNode.next
    return firstNode

'''
118. Pascal's Triangle
建立好2d list往里填充即可
'''

    def generate(self, numRows: int) -> List[List[int]]:

        triangle =[[1 for _ in range(j + 1)] for j in range(numRows)]
        
        for i in range(1, numRows):

            for j in range(1, i + 1):
                if j == i:
                    continue
                else:
                    triangle[i][j] = triangle[i - 1][j - 1] + triangle[i - 1][j]

        return triangle

'''
119. Pascal's Triangle II
可以只用一个list，计算每一行时向 0 index加入1来方便计算
错误点： 每一行的最后一位不需要计算
'''
def getRow(self, rowIndex: int) -> List[int]:
    
    row = []
        
    for i in range(rowIndex + 1):
        row.insert(0, 1)
        for j in range(1, len(row) - 1):
            row[j] = row[j] + row[j + 1]
    return row




'''
125. Valid Palindrome
简单two pointer
'''
def isPalindrome(self, s: str) -> bool:
    if not s or len(s) == 0:
        return True
    
    if len(s) == 1:
        return True

    i, j = 0, len(s) - 1
    
    while i < j:
        while i < j and not s[i].isalpha() and not s[i].isnumeric():
            i += 1
        while i < j and not s[j].isalpha() and not s[j].isnumeric():
            j -= 1
        if i < j and s[i].lower() != s[j].lower():
            return False
        i += 1
        j -= 1
            
    return True

'''
<<<<<<< HEAD
129. Sum Root to Leaf Numbers
'''

def sumNumbers(self, root: TreeNode) -> int:
    if not root:
        return 0
    return self.helper(root, 0)
    
def helper(self, root, curr):
    if not root:
        return 0
    if not root.left and not root.right:
        return curr *10 + root.val
    
    val = curr * 10 + root.val
    
    left = self.helper(root.left, val)
    right = self.helper(root.right, val)
    
    return left + right

'''
130. Surrounded Regions
联通块，从边上开始向内搜索
'''

def solve(self, board: List[List[str]]) -> None:
"""
Do not return anything, modify board in-place instead.
"""
if not board or len(board) == 0:
    return

n, m = len(board), len(board[0])

for i in range(n):
    self.dfs(i, 0, board)
    self.dfs(i, m - 1, board)

for j in range(m):
    self.dfs(0, j, board)
    self.dfs(n - 1, j, board)
    
    
for i in range(n):
    for j in range(m):
        if board[i][j] == 'G':
            board[i][j] = 'O'
        else:
            board[i][j] = 'X'
        
        
def dfs(self, x, y, board):
    n, m = len(board), len(board[0])
    
    if x < 0 or x >= n or y < 0 or y >= m:
        return
    if board[x][y] != 'O':
        return
    if board[x][y] == 'G':
        return 
    board[x][y] = 'G'            
    self.dfs(x + 1, y, board)
    self.dfs(x, y + 1, board)
    self.dfs(x - 1, y, board)
    self.dfs(x, y - 1, board)


'''
131. Palindrome Partitioning
'''
def partition(self, s: str) -> List[List[str]]:
    if not s or len(s) == 0:
        return []
    
    res = []
    self.helper(s, 0, [], res)
    return res

def helper(self, s, start, curr, res):
    if start >= len(s):
        res.append(curr[:])
        return
    
    for i in range(start, len(s)):
        prefix = s[start : i + 1]
        if not self.isPalindrome(prefix):
            continue
        curr.append(prefix)
        self.helper(s, i + 1, curr, res)
        curr.pop()
    
def isPalindrome(self, s):
    if not s or len(s) == 0:
        return False
    start, end = 0, len(s) - 1
    while start < end:
        if s[start] != s[end]:
            return False
        start += 1
        end -= 1
    return True


'''
133. Clone Graph
解法：bfs 建立所有的点，用map把新点和对应的旧点链接
然后用一个forloop来链接新点之间的联系
'''
def cloneGraph(self, node: 'Node') -> 'Node':
    if not node or node is None:
        return

    queue = collections.deque([node])
    mapping = {}
    
    while queue:
        point = queue.popleft()
        if point not in mapping:
            newNode = Node(point.val)
            mapping[point] = newNode
            for neighbor in point.neighbors:
                queue.append(neighbor)
    
    for point in mapping:
        for neighbor in point.neighbors:
            mapping[point].neighbors.append(mapping[neighbor])
            
    return mapping[node]

'''
134. Gas Station
解法：O(n)
用curr记录当前起点的总油量，若小于0则无法以当前起点到i为起点, 因为油量不够.
重新以i + 1为起点, 记录该起点油量.
另记录总油量来判断是否能走完全程.
'''
def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:       
    totalGas, curr = 0, 0
    for i in range(len(gas)):
        curr += gas[i] - cost[i]
        if curr < 0:
            start = i + 1
            curr = 0
        
        totalGas += curr
        
    if totalGas < 0:
        return -1
    else:
        return start
'''
139. Word Break
dfs + memo： 注意取prefix的边界
'''
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    memo = {}
    return self.helper(s, 0, wordDict, memo)

def helper(self, s, startIndex, wordDict, memo):
    if startIndex in memo:
        return memo[startIndex]
    
    if startIndex >= len(s):
        return True
    
    for i in range(startIndex, len(s)):
        prefix = s[startIndex : i + 1]
        
        if prefix not in wordDict:
            continue
        if self.helper(s, i + 1, wordDict, memo):
            memo[startIndex] = True
            return True
    memo[startIndex] = False 
    return False

'''
140. Word Break II
dfs + memo
解法： 把合法的prefix加入到剩余的string的答案中
注意当 start >= len(s)时返回[""]， 这种情况res.append(prefix)
'''

def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
    memo = {}
    return self.helper(s, 0, wordDict, memo)
    

def helper(self, s, start, wordDict, memo):
    if start in memo:
        return memo[start]
    
    if start >= len(s):
        return [""]
    res = []
    
    for i in range(start, len(s)):
        prefix = s[start : i + 1]
        if prefix not in wordDict:
            continue
        rightRes = self.helper(s, i + 1, wordDict, memo)
        for right in rightRes:
            if right != "":
                res.append(prefix + " " + right)
            else:
                res.append(prefix)
    
    memo[start] = res
    return res

'''
142. Linked List Cycle II
规律： 先找相遇点，再用一个pointer从头开始走，想遇到的点就是回转点
'''
def detectCycle(self, head: ListNode) -> ListNode:
    if not head:
        return None
    
    dummy = ListNode(0)
    p1 = p2 = p3 = dummy
    dummy.next = head
    
    while p1 and p2 and p2.next:
        p1 = p1.next
        p2 = p2.next.next
        if p1 == p2:
            break
    
    
    while p2:
        p2 = p2.next
        p3 = p3.next
        if p2 == p3:
            return p3
    
    return None

'''
143. Reorder List
解法： 平分成两段，reverse后一段再重新连接
'''

def reorderList(self, head: ListNode) -> None:
    """
    Do not return anything, modify head in-place instead.
    """
    if not head:
        return None
    dummy = ListNode(0)
    dummy.next = head
    p1 = p2 = dummy
    pre = None
    while p2 and p2.next:
        pre = p1
        p1 = p1.next
        p2 = p2.next.next
    h1 = head
    h2 = self.reverse(p1.next)
    p1.next = None
    self.combine(h1, h2)
    return head
    
def reverse(self, head):
    if not head:
        return None
    p1, p2 = head, head.next
    
    while p2:
        temp = p2.next
        p2.next = p1
        p1 = p2
        p2 = temp
    
    head.next = None
    return p1
        
def combine(self, h1, h2):
    while h1 and h2:
        temp = h1.next
        h1.next = h2
        h1 = temp
        h1, h2 = h2, h1

'''
144. Binary Tree Preorder Traversal
'''
def preorderTraversal(self, root: TreeNode) -> List[int]:
    if not root:
        return []
    res = []
    self.traverse(root, res)
    return res

def traverse(self, root, res):
    if not root:
        return 
    
    res.append(root.val)
    self.traverse(root.left, res)
    self.traverse(root.right, res)

'''
147. Insertion Sort List
注意理解insertion是从头开始找比当前选中元素大的元素
'''
def insertionSortList(self, head: ListNode) -> ListNode:
    if not head:
        return head
    dummy = ListNode(0)
    dummy.next = head
    curr = head
    
    while curr and curr.next:
        if curr.val < curr.next.val:
            curr = curr.next
        else:
            pre = dummy
            temp = curr.next
            curr.next = curr.next.next
            while pre.next.val < temp.val:
                pre = pre.next
            temp.next = pre.next
            pre.next = temp
        
    return dummy.next

'''
148. Sort List
解法：merge sort
分割list成两段，再进行merge
注意： sortList出口条件是当head.next没有时返回当前的node
'''

def sortList(self, head: ListNode) -> ListNode:
    if not head or not head.next:
        return head
    dummy = ListNode(0)
    dummy.next = head
    
    slow, fast = dummy, dummy
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    part1 = dummy.next
    part2 = slow.next
    slow.next = None
    
    return self.merge(self.sortList(part1), self.sortList(part2))

def merge(self, p1, p2):
    dummy = ListNode(0)
    tail = dummy
    while p1 and p2:
        if p1.val < p2.val:
            tail.next = p1
            p1 = p1.next
        else:
            tail.next = p2
            p2 = p2.next
        tail = tail.next
    if p1:
        tail.next = p1
    if p2:
        tail.next = p2
    return dummy.next

'''
152. Maximum Product Subarray
需要记录当前最小值
'''
def maxProduct(self, nums: List[int]) -> int:
    if not nums or len(nums) == 0:
        return 0
    
    n = len(nums)
    currMax = nums[0]
    currMin = nums[0]
    product = nums[0]
    
    for i in range(1, n):
        currMax, currMin = max(currMax * nums[i], nums[i], currMin * nums[i]), min(currMax * nums[i], nums[i], currMin * nums[i])
        product = max(currMax, product)
    return product

'''
153. Find Minimum in Rotated Sorted Array
解法： 二分法排除不可能的区间
'''

'''
154. Find Minimum in Rotated Sorted Array II
多考虑一个如何去重的问题
'''
def findMin(self, nums: List[int]) -> int:        
    if not nums or len(nums) == 0:
        return -1
    start, end = 0, len(nums) - 1
    
    while start + 1 < end:
        mid = (start + end) // 2
        if nums[mid] < nums[end]:
            end = mid
        elif nums[mid] > nums[end]:
            start = mid
        else:
            while end > mid and nums[end] == nums[mid]:
                end -= 1
    
    return min(nums[start], nums[end])