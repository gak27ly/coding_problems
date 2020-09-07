#leetcode 
# 1, 2, 3, 5, 6, 7, 9, 11, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 26, 27, 
# 28, 29, 31, 33, 34, 35, 36, 38, 39, 40, 41, 43, 46, 47, 48 

'''
需重做： 6, 5, 19, 20, 22, 23, 
26 -48
'''


'''
15. 3Sum
错误处1： 选择一个固定数字时没有对接下来2sum的范围进行规定。
因该是规定在i + 1 到len(nums)的范围内进行查找，否则答案中会有重复 [[-1, 0 ,1], [0, -1, 1]]
进行规定后，[0, -1, 1]就不会被选中， 因为选定0后不会再往前看到 -1

错误出2: twoSum中去重只需要在 ==target的时候用一个while loop找到下一个不相等的点就可以了
'''

'''
16. 3Sum Closest
注意点： 可以直接在 i in range(len(nums) - 2)中确定一个点
再在 i + 1 到len(nums) - 1中去找两个点来相加 = total
用total和target比较并记录最小的差值
'''

'''
19. Remove Nth Node From End of List
去除倒数第n个node
解法：利用dummy node从头开始使fast node与slow node保持n+1的距离
然后让slow node跳过下一个node就能完成删除的操作
'''
def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
    dummy = ListNode()
    slow = fast = dummy
    dummy.next = head
    for _ in range(n + 1):
        fast = fast.next
    
    
    while fast:
        fast = fast.next
        slow = slow.next
    
    slow.next = slow.next.next

    return dummy.next

'''
22. Generate Parentheses
注意点：因为只需要考虑加左边还是加右边，可以直接调用两个dfs function, 
若有多种选择则需要用for loop来选中一种再dfs并去backtracking
用left和right关系来判断是否是否合法
'''
def generateParenthesis(self, n: int) -> List[str]:
    if n == 0:
        return []
    res = []
    self.dfs(n, 0, 0, res, "")  
    return res


def dfs(self, n, left, right, res, combination):
    if right > left:
        return
    if left > n:
        return
    
    if left == n and right == n:
        res.append(combination)
        return 
    
    self.dfs(n, left + 1, right, res, combination + ("("))
    
    self.dfs(n, left, right + 1, res, combination + (")"))


'''
23. Merge k Sorted Lists
注意点： 使用index来划分传入的list of nodes 来减少空间复杂度
'''

'''
24. Swap Nodes in Pairs
每两个node交换位置
注意点： loop的边界情况
利用更清晰的变量名称来帮助思考,完成一轮交换位置后，
将变量按位置向后移动
'''
def swapPairs(self, head):
    # write your code here
    if head is None or head.next is None:
        return head
    
    prev = ListNode(0, head)
    dummy = prev
    
    while head and head.next:
        
        secondnode = head.next
        temp = secondnode.next
        
        prev.next = secondnode
        secondnode.next = head
        head.next = temp
        
        prev = head
        head = temp
        
    return dummy.next

'''
28. strstr: 找到string中needle string最先出现的位置
解法： 在string中按照起始点查找i: i + len(needle)
time: O(n)
'''

def strStr(self, haystack: str, needle: str) -> int:
    if not needle:
        return 0
    
    n, m = len(haystack), len(needle)
    if m == 0:
        return 0
    
    if n < m:
        return -1
    
    for i in range(n - m + 1):
        if haystack[i: i + m] == needle:
            return i
    
    return -1

'''
31. Next Permutation
解： 从后向前找到第一个非递增的位置i，再从后向前找到第一个比nums[i]大的数字，
交换他们的位置，对i之后的数字首位交换位置以得到从i+1开始最小的顺序
time: O(n)
'''
def nextPermutation(self, nums: List[int]) -> None:
    if len(nums) < 2:
        return nums
    
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
    
    if i != -1:
        for j in range(len(nums) - 1, i, -1):
            if nums[j] <= nums[i]:
                continue
            nums[j], nums[i] = nums[i], nums[j]
            break
    
    start = i + 1
    end = len(nums) - 1
    #对i之后重新排序，因为从len(nums) - 1到i + 1是递增序列，交换首尾得到最小序列
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1
    return nums
'''
33. Search in Rotated Sorted Array
注意点：二分法的结束条件 start + 1 < end
避免产生 [1, 2] 中mid只能取到start的情况而产生死循环
time: O(logn)
'''
def search(self, nums: List[int], target: int) -> int:
    if not nums or len(nums) == 0:
        return -1
    
    start, end = 0, len(nums) - 1
    
    while start + 1 < end:
        mid = (start + end) // 2
        
        if nums[mid] <=  nums[end]:
            if  nums[mid] <= target <= nums[end]:
                start = mid
            else:
                end = mid
        elif nums[start] <= nums[mid]:
            if nums[start] <= target <= nums[mid]:
                end = mid
            else:
                start = mid 
    if nums[start] == target:
        return start
    if nums[end] == target:
        return end
    return -1
'''
34. Find First and Last Position of Element in Sorted Array
解法： 做两遍binary search 分别找到第一个target和最后一个target
错误点： 分析start end 哪一个是要找的点时，要用if elif来保证正确的答案.

'''
def searchRange(self, nums: List[int], target: int) -> List[int]:
    if not nums:
        return [-1, -1]
    
    start, end = 0, len(nums) - 1
    
    while start + 1 < end:
        mid = (start + end) // 2
        
        if nums[mid] <= target:
            start = mid
        else:
            end = mid
            
    if nums[end] == target:
        last = end
    #如果用if statement 相同的值会使last的位置改变
    elif nums[start] == target:
        last = start
    else:
        return [-1, -1]
    
    start, end = 0, len(nums) - 1 
    while start + 1 < end:
        mid = (start + end) // 2
        if nums[mid] < target:
            start = mid
        else:
            end = mid
    if nums[start] == target:
        first = start
    elif nums[end] == target:
        first = end
    else:
        return [-1, -1]
    
    return [first, last]


'''
41. First Missing Positive
数字范围1 - n， 找到唯一的重复数字
利用array的特点， 把存在的数字对应的index位置的数值改为负值
找到第一个非负的值，返回其index + 1
'''

def firstMissingPositive(self, nums: List[int]) -> int:
    flag = False
    for i in range(len(nums)):
        if nums[i] == 1:
            flag = True
    
    if not flag:
        return 1
    
    for i in range(len(nums)):
        if nums[i] <= 0:
            nums[i] = 1
    
    for i in range(len(nums)):
        if abs(nums[i]) > len(nums):
            continue
        if abs(nums[i]) == len(nums):
            nums[0] = -abs(nums[0])
        else:
            nums[abs(nums[i])] = -abs(nums[abs(nums[i])])
        
    print(nums)
    for i in range(1, len(nums)):
        if nums[i] > 0:
            return i
    
    if nums[0] > 0:
        return len(nums)
    
    return len(nums) + 1
'''
42. Trapping Rain Water
解法1: dp，建立两个数组来记录位置i左边的最高挡板和右边的最高挡板
i位置的水量 = 较低的挡板高度 - i位置挡板的高度
'''
def trap(self, height: List[int]) -> int:
    n = len(height)
    if not height or n == 0:
        return 0
    
    l = [0 for _ in range(n)]
    r = [0 for _ in range(n)]
    
    l[0] = height[0]
    r[n - 1] = height[n - 1]
    
    for i in range(1, n):
        l[i] = max(height[i], l[i - 1])
    
    for i in range(n - 2, -1, -1):
        r[i] = max(height[i], r[i + 1])
    
    res = 0
    for i in range(n):
        res += min(l[i], r[i]) - height[i]
        
    return res
'''
解法2: 通过two pointer, 用lMax,rMax来代表左边和右边的最高挡板高度，
哪一边的挡板底就把那一边的水量加入res，水量 = 底挡板高度 - 该位置高度
然后移动l,r指针，并更新对应的lMax or rMax
'''
def trap(self, height: List[int]) -> int:
    n = len(height)
    if not height or n == 0:
        return 0
    
    l, r = 0, n - 1
    lMax, rMax = height[l], height[r]
    res = 0
    
    while l < r:
        if lMax < rMax:
            res += lMax - height[l]
            l += 1
            lMax = max(lMax, height[l])
        else:
            res += rMax - height[r]
            r -= 1
            rMax = max(rMax, height[r])
    
    return res

'''
43. Multiply Strings

建立长度为 n + m 的array，使用array的位置来模拟并存储乘法的结果
'''

def multiply(self, num1: str, num2: str) -> str:
    res = [0] * (len(num1) + len(num2))
    
    for i in range(len(num1) - 1, -1, -1):
        for j in range(len(num2) - 1, -1, -1):
            x = ord(num2[j]) - ord('0') 
            y = ord(num1[i]) - ord('0')
            
            res[i + j], res[i + j + 1] = (res[i + j] * 10 + res[i + j + 1] + x * y) // 10,  (res[i + j + 1] + x * y) % 10
                         
    while res[0] == 0 and len(res) > 1:
        res.pop(0)
            
    for i in range(len(res)):
        res[i] = chr(res[i] + 48)
    return "".join(res)


'''
48. Rotate Image
将行变成列，然后把列的位置进行调整
不管是向左还是向右旋转90度，都会导致每一行变成列。
'''
def rotate(self, matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    n, m = len(matrix), len(matrix[0])
    
    for i in range(n): 
        for j in range(i , m):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
            
    for i in range(n):
        start, end = 0, len(matrix[0]) - 1
        while start < end:
            
            matrix[i][start], matrix[i][end] = matrix[i][end], matrix[i][start]
            start += 1
            end -= 1
    
    return matrix

'''
49. Group Anagrams
解法： 利用dafalutdict 
把每个单词sorted成一个list后一tuple的形式作为key存入dict，把这单词append到这个key下的list中
mapping.values()就是所有存入的单词的集合
'''
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    
    if not strs or len(strs) == 0:
        return [[]]
    
    mapping = collections.defaultdict(list)
    
    for chars in strs:
        mapping[tuple(sorted(chars))].append(chars)
    
    return mapping.values()

'''
50. Pow(x, n)
解法： 当n为奇数时，将当前base * res 来保证不会少乘
然后继续对base进行倍增 知道n为1
'''
def myPow(self, x: float, n: int) -> float:
    if x == 0 or n == 0:
        return 1
    
    if n < 0:
        x = 1 / x
        n = -n
    res = 1
    num = x
    
    while n > 1:
        if n % 2 == 1:
            res *= num

        num = num * num
        n = n // 2 
        
    res *= num
    return res

'''
52. N-Queens II
解法： 利用两个array:dia antiDia来记录两个对角线是否被使用
注意点： count不能用int，应为pass by value
放在list中才能返回正确的值

'''
def totalNQueens(self, n: int) -> int:
    count = [0]
    dia = [True for _ in range(2 * n - 1)]
    antiDia = [True for _ in range(2 * n - 1)]
    self.helper(n, [], count, dia, antiDia)
    return count[0]

def helper(self, n, rows, count, dia, antiDia):
    if len(rows) == n:
        count[0] += 1
        return
    
    row = len(rows) 
    for i in range(n):
        if dia[i + row] and antiDia[n - (row - i) - 1] and i not in rows:
            dia[i + row], antiDia[n - (row - i) - 1] = False, False                
            rows.append(i)
            self.helper(n, rows, count, dia, antiDia)
            rows.pop()
            dia[i + row], antiDia[n - (row - i) - 1] = True, True

'''
54. Spiral Matrix
确定上下左右边界，和方向坐标，通过缩小上下左右范围和改变方向来遍历所有的点。
'''

'''
55. Jump Game 
贪心算法， 看n-2的位置能不能跳到 n-1，如果可以看接着向前看从哪能跳到n-2
'''
'''
56. Merge Intervals
排序后判断是否overlap，有的话扩展右边界， 没有就加入答案.
'''
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    if not intervals or len(intervals) == 0:
        return []
    intervals = sorted(intervals, key = lambda x : x[0])
    res = []
    
    left = intervals[0][0]
    right = intervals[0][1]
    
    print(intervals)
    for i in range(1, len(intervals)):
        
        if right < intervals[i][0]:
            res.append([left, right])
            left = intervals[i][0]
            right = intervals[i][1]
        else:
            right = max(intervals[i][1], right)
            
    res.append([left, right])
    
    return res
'''
58. Length of Last Word
找到最后一个char的位置，再找最后一个space的位置，注意找不到这个位置的情况。
'''

'''
62. Unique Paths
'''
def uniquePaths(self, m: int, n: int) -> int:
    
    dp = [[0 for _ in range(m)] for _ in range(2)]
    
    for i in range(n):
        for j in range(m):
            if i == 0 or j == 0:
                dp[i % 2][j] = 1
            else:
                dp[i % 2][j] = dp[i % 2 - 1][j] + dp[i % 2][j - 1]
                
    return dp[n % 2 - 1][m - 1]
'''
63. Unique Paths
先把第一行第一列进行预处理.
'''

'''
64. Minimum Path Sum
dp 对第一行第一列进行预处理.
'''
def minPathSum(self, grid: List[List[int]]) -> int:
    if not grid:
        return 0
    
    n = len(grid)
    m = len(grid[0])
    
    dp = [[0 for _ in range(m)] for _ in range(n)]
    dp[0][0] = grid[0][0]
    
    for row in range(1, n):
        dp[row][0] = grid[row][0] + dp[row - 1][0]
    
    for col in range(1, m):
        dp[0][col] = grid[0][col] + dp[0][col - 1]
    
    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
    
    return dp[n - 1][m - 1]

#利用滚动数字减少空间复杂度， 预处理第一行即可
def minPathSum(self, grid: List[List[int]]) -> int:
    if not grid:
        return 0
    
    n = len(grid)
    m = len(grid[0])
    
    dp = [[0 for _ in range(m)] for _ in range(2)]
    dp[0][0] = grid[0][0]
    
    
    for col in range(1, m):
        dp[0][col] = grid[0][col] + dp[0][col - 1]
    
    for i in range(1, n):
        for j in range(m):
            if j > 0:
                dp[i % 2][j] = grid[i][j] + min(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
            else:
                dp[i % 2][j] = grid[i][j] + dp[(i - 1) % 2][j]
    # print(dp)
    return dp[(n - 1) % 2][m - 1]
'''
67. Add Binary
错误点：循环开头要把carry赋值给total而不是累加到total
'''
def addBinary(self, a: str, b: str) -> str:

    res = 0
    i, j = len(a) - 1, len(b) - 1 
    carry = 0
    
    total = 0
    res = ""
    while i >= 0 or j >= 0:
        total = carry
        if i >=0:
            total += int(a[i])
        if j >=0:
            total += int(b[j])
        res += str(total % 2)
        carry = total // 2
        i -= 1
        j -= 1
        
    if carry == 1:
        res += str(carry)
        
    return res[::-1]

'''
69. Sqrt(x)
binary search
找到第一个在 1-x/2范围内 n ** 2 小于等于x的数
注意x = 0，1 的特殊情况
'''

'''
70. Climbing Stairs
注意 input为0， 1的边界情况
'''

'''
73. Set Matrix Zeroes
从 1，1开始遍历并利用第一行和第一列的值来标记该行该列是否应该被改为0
对第一行和第一列单独进行遍历，并用两个变量来记录其是否为0
注意点：先对除了第一行第一列的行和列进行变化，然后再变化第一行第一列
否则会改变正确的标记
'''

def setZeroes(self, matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    if not matrix or len(matrix) == 0:
        return matrix
    
    n = len(matrix)
    m = len(matrix[0])
    
    firstRow = False
    firstCol = False
    
    for row in range(n):
        if matrix[row][0] == 0:
            firstCol = True
            break
    
    for col in range(m):
        if matrix[0][col] == 0:
            firstRow = True
            break
    
    for i in range(1, n):
        for j in range(1, m):
            if matrix[i][j] != 0:
                continue
            matrix[0][j] = 0
            matrix[i][0] = 0

    #根据上面的标记对 1- n-1行进行改变
    for row in range(1, n):
        if matrix[row][0] != 0:
            continue
        for col in range(m):
            matrix[row][col] = 0

    #根据上面的标记对 1- m-1列进行改变
    for col in range(1, m):
        if matrix[0][col] != 0:
            continue
        for row in range(n):
            matrix[row][col] = 0
            
    if firstRow == True:
        for col in range(m):
            matrix[0][col] = 0

    if firstCol == True:
        for row in range(n):
            matrix[row][0] = 0

    return matrix

'''
74. Search a 2D Matrix
利用两个binary search
错误点：没有考虑到[[]] 的边界条件, 
第一个binary search结束时，没考虑matrix[end][0] == target的情况
'''
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    if not matrix or len(matrix) == 0 or len(matrix[0]) == 0:
        return False

    
    n = len(matrix)
    m = len(matrix[0])
    
    start, end = 0, n - 1
    
    while start + 1 < end:
        mid = start + (end - start) // 2
        if matrix[mid][0] > target:
            end = mid
        elif matrix[mid][0] < target:
            start = mid
        else:
            return True
    
    if matrix[end][0] <= target:
        return self.searchRow(matrix, end, target)
    else:
        return self.searchRow(matrix, start, target)
     
def searchRow(self, matrix, row, target):
    start, end = 0, len(matrix[row]) - 1
    
    while start + 1 < end:
        mid = start + (end - start) // 2
        if matrix[row][mid] > target:
            end = mid
        elif matrix[row][mid] < target:
            start = mid
        else:
            return True
    
    if matrix[row][start] == target:
        return True
    if matrix[row][end] == target:
        return True
    return False

'''
75. Sort Colors
错误点：left pointer的点不会丢给middle一个为0的点，
但需要考虑right pointer丢给middle一个为0或2的点的情况
'''
def sortColors(self, nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    if not nums or len(nums) < 2:
        return
    l, r = 0, len(nums) - 1
    middle = 0
    
    while middle <= r:
        if nums[middle] == 0:
            nums[l], nums[middle] = nums[middle], nums[l]
            l += 1
            middle += 1
        elif nums[middle] == 2:
            nums[r], nums[middle] = nums[middle], nums[r]
            r -= 1
            if nums[middle] == 1:
                middle += 1
        else:
            middle += 1
'''
77. Combinations
78. Subsets
区别是78题需要把所有的组合加入result中，77只需要加入给定长度为k的组合
'''

'''
79. Word Search
使用dfs，可以将board的元素暂时变为0，来表示此点已经visit过。
递归搜索完成后再改回来.
'''
def exist(self, board: List[List[str]], word: str) -> bool:
    if not word or len(word) == 0:
        return True
    
    if not board:
        return False
    
    queue = collections.deque()
    
    
    n = len(board)
    m = len(board[0])
    visited = []
    for i in range(n):
        for j in range(m):
            if self.dfs(i, j, board, word, 0):
                return True
    return False

def dfs(self, x, y, board, word, index):
    DIRECTIONS = [(1,0), (-1, 0), (0, 1), (0, -1)]
    if x < 0 or x > len(board) - 1 or y < 0 or y > len(board[0]) - 1:
        return False
    if board[x][y] != word[index]:
        return False
    
    if index == len(word) - 1:
        return True
    
    temp = board[x][y]
    board[x][y] = 0
    found =  (self.dfs(x + 1, y, board, word, index + 1) or 
            self.dfs(x - 1, y, board, word, index + 1) or 
            self.dfs(x, y + 1, board, word, index + 1) or 
            self.dfs(x, y - 1, board, word, index + 1))
    
    board[x][y] = temp
    return found


'''
81. Search in Rotated Sorted Array II
注意点： 需要调整end的位置来保证mid和end不是相同元素，
否则无法确定array的哪一部分是正常排序的
'''
def search(self, nums: List[int], target: int) -> bool:
    
    if not nums:
        return False
    
    start, end = 0, len(nums) - 1

    while start + 1 < end:

        mid = start + (end - start) // 2
        
        if nums[mid] == target:
            return True
        
        if nums[mid] < nums[end]:
            if nums[mid] <= target <= nums[end]:
                start = mid
            else:
                end = mid
        elif nums[mid] > nums[end]:
            if nums[start] <= target <= nums[mid]:
                end = mid
            else:
                start = mid
        else:
            end -= 1
    
    return nums[start] == target or nums[end] == target
'''
86. Partition List
Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
错误点：需要用两个dummy node, 如果part1，part2是同一个dummy node会导致part1或part2指向错误的node
'''

def partition(self, head: ListNode, x: int) -> ListNode:
    if not head:
        return None
    temp = head
    dummy1 = ListNode(0)
    dummy2 = ListNode(0)
    part1, part2 = dummy1, dummy2

    
    while temp:
        if temp.val < x:
            part1.next = temp
            part1 = part1.next
        else:
            part2.next = temp
            part2 = part2.next
        temp = temp.next
        
    part2.next = None
    part1.next = dummy2.next
    
    return dummy1.next

'''
88. Merge Sorted Array
解法：因为需要把所有数字并入nums1，所以可以先把最大的数字放到nums1最后
最后如果nums2还有剩余数字，就从后向前放入nums1的前面
'''


'''
95. Unique Binary Search Trees II

'''
#方法一
def generateTrees(self, n: int) -> List[TreeNode]:
        if n == 0:
            return []
        return self.helper(1, n)
    
def helper(self, start, end):
    res = []
    if start > end:
        return res
    
    for i in range(start, end + 1):
        leftStructs = self.helper(start, i - 1)
        rightStructs = self.helper(i + 1, end)
        
        if len(leftStructs) == 0 and len(rightStructs) == 0:
            head = TreeNode(i)
            res.append(head)
            
        elif len(leftStructs) == 0:
            for rightNode in rightStructs:
                head = TreeNode(i)
                head.right = rightNode
                res.append(head)
        elif len(rightStructs) == 0:
            for leftNode in leftStructs:
                head = TreeNode(i)
                head.left = leftNode
                res.append(head)
        else:
            for leftNode in leftStructs:
                for rightNode in rightStructs:
                    head = TreeNode(i)
                    head.right = rightNode
                    head.left = leftNode
                    res.append(head)
    return res

#方法二
def generateTrees(self, n: int) -> List[TreeNode]:
    if n == 0:
        return []
    return self.helper(1, n)

def helper(self, start, end):
    res = []
    if start > end:
        res.append(None)
        return res
    
    for i in range(start, end + 1):
        leftStructs = self.helper(start, i - 1)
        rightStructs = self.helper(i + 1, end)
        
        for leftNode in leftStructs:
            for rightNode in rightStructs:
                head = TreeNode(i)
                head.left = leftNode
                head.right = rightNode
                res.append(head)
    return res


'''
96. Unique Binary Search Trees
解法：使用dp，j代表的是左边子数节点的数量
dp[i]就是dp[左子数节点数量] * dp[右子树节点数量]
'''
def numTrees(self, n: int) -> int:
    
    dp = [0 for _ in range(n + 1)]
    
    dp[0] = 1
    dp[1] = 1
    
    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - j - 1]

    return dp[n]



'''
98. Validate Binary Search Tree
利用上下边界 自顶向下检验左子树和右子树
'''
def isValidBST(self, root: TreeNode) -> bool:
    
    if not root:
        return True
    minVal = -sys.maxsize - 1
    maxVal = sys.maxsize
    
    return self.helper(root, minVal, maxVal)

def helper(self, root, minVal, maxVal):
    if not root:
        return True
    
    if root.val <= minVal or root.val >= maxVal:
        return False
    return self.helper(root.left, minVal, root.val) and self.helper(root.right, root.val, maxVal)



