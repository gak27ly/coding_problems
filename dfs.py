"""
17. Subsets
Given a set of distinct integers, return all possible subsets.
@param nums: A set of numbers
@return: A list of lists
"""
def subsets(self, nums):
    # write your code here
    
    if not nums:
        return [[]]
        
    nums.sort()
    
    res = []
    self.helper(nums, [], res)
    return res

def helper(self, nums, combination, res):
    res.append(list(combination))
    
    for i in range(len(nums)):
        prefix = nums[i]
        combination.append(prefix)
        self.helper(nums[i + 1 :], combination, res)
        combination.pop()
'''
使用index来标记当前数字，节省空间
出错处：在helper中使用了index + 1导致出错，应该使用i + 1
'''

def subsets(self, nums):
    # write your code here
    if not nums:
        return [[]]
        
    nums.sort()
    
    res = []
    self.helper(nums, 0, [], res)
    return res

def helper(self, nums, index, combination, res):
    res.append(list(combination))
    
    for i in range(index, len(nums)):
        combination.append(nums[i])
        self.helper(nums, i + 1, combination, res)
        combination.pop()

'''
Permutations II
Given a list of numbers with duplicate number in it. 
Find all unique permutations.
解法：利用visited来判断是否该数字已经取过了
需要先排序把重复的数字放到一起来判断取一个数字为起始点时是否与前面数字一样
这种情况需要continue   
'''
def permuteUnique(self, nums):
    # write your code here
    if not nums:
        return [[]]
        
    nums.sort()
    res = []
    visited = [False for _ in range(len(nums))]

    self.helper(nums, visited, [], res)
    return res
    
def helper(self, nums, visited, premutation, res):
    if len(premutation) == len(nums):
        res.append(premutation[:])
        
    lastItem = None
    for i in range(len(nums)):
        if visited[i] == True:
            continue
        if lastItem == nums[i]:
            continue
        premutation.append(nums[i])
        visited[i] = True
        self.helper(nums, visited, premutation, res)
        lastItem = premutation.pop()
        visited[i] = False


"""
求当前permutation是从小到大第几个permutation
@param A: An array of integers
@return: A long integer
注意点： 从后往前来积累factorial的数值
"""
def permutationIndex(self, A):
    # write your code here
    if not A:
        return 1
    n = len(A)
    fact = 1
    index = 0
    
    for i in range(n - 1, -1, -1):
        count = 0
        for j in range(i + 1, n):
            if A[j] < A[i]:
                count += 1
        index += count * fact    
        fact *= n - i
    
    return index + 1
'''
求当前permutation是从小到大第几个permutation
数组含有重复元素
解法： 利用repeat 这个hashmap来记录从后往前有多少重复元素，并记录他们的乘积
计算当前位置的排列数时除以当前重复元素数量的factorial

'''
def permutationIndexII(self, A):
    # write your code here
    
    if not A:
        return 1
    n = len(A)
    index = 0
    fact = 1
    repeat = {}
    repeatFact = 1
    
    for i in range(n - 1, -1, -1):
        if A[i] in repeat:
            repeat[A[i]] += 1
        else:
            repeat[A[i]] = 1
            
        repeatFact *= repeat[A[i]]
        
        smaller = 0
        for j in range(i + 1, n):
            if A[j] < A[i]:
                smaller += 1
        
        index += smaller * fact // repeatFact
        fact *= (n - i)
        
    return index + 1




def longestIncreasingPath(matrix):
    if not matrix:
        return 0
    n = len(matrix)
    m = len(matrix[0])
    ans = 0
    for i in range(n):
        for j in range(m):
            ans = max(helper(matrix, i, j, []), ans)
    return ans
    
def helper(matrix, x, y, visited):
    steps = 1
    DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if (x, y) not in visited:
        visited.append((x, y))
    
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if isValid(matrix, nx, ny) and (nx, ny) not in visited and \
            matrix[nx][ny] - matrix[x][y] > 3:
            steps = max(helper(matrix, nx, ny, visited) + 1, steps)
            
    visited.pop()
    
    return steps

def isValid(matrix, x, y):
    if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]):
        return True
    return False
    
    
A = [[1, 2, 3], [4, 5, 6],[7, 8, 9]]
print(longestIncreasingPath(A))




'''
512. Decode Ways
A message containing letters from A-Z is being 
encoded to numbers using the following mapping:
'A' -> 1
'B' -> 2
'''

def numDecodings(s):

    if not s or len(s) == 0:
        return []
        
    n = len(s)
    
    res = []
    
    dfs(s, [], res)
    res = sorted(res, key = len)
    return res
        
def dfs(s, decode, res):
    if len(s) == 0:
        res.append("".join(decode))
        return 
    
    if s[0] == "0":
        return

    for i in range(1, len(s) + 1):
        prefix = s[:i]
        if prefix[0] == "0":
            continue
        if int(prefix) > 26:
            break
        
        decode.append(chr(int(prefix) + 96))
        dfs(s[i:], decode, res)
        decode.pop()
            
print(numDecodings("105"))


'''
90. k Sum II
Given n unique postive integers, number k (1<=k<=n) and target.
Find all possible k integers where their sum is target.
注意点： 此题没有去重的问题, 题目默认已经排好序。 在pathSum 基础上添加了k这个条件
实际就是求所有组合中长度是k并满足sum是target的组合
'''
def kSumII(self, A, k, target):
    # write your code here
    if len(A) < k:
        return []
    res = []
    
    self.helper(A, k, 0, target, [], res)
    return res

def helper(self, A, k, start, target, combination, res):
    if len(combination) > k:
        return 
    
    if target < 0:
        return
    
    if len(combination) == k and target == 0:
        res.append(list(combination))
        return
    
    for i in range(start, len(A)):
        combination.append(A[i])
        self.helper(A, k, i + 1, target - A[i], combination, res)
        combination.pop()
'''
33. N-Queens
中文English
The n-queens puzzle is the problem of placing n queens on an n×n chessboard 
such that no two queens attack each other(Any two queens can't be in the same row, column, diagonal line).
Given an integer n, return all distinct solutions to the n-queens puzzle.
Each solution contains a distinct board configuration of the n-queens' placement, 
where 'Q' and '.' each indicate a queen and an empty space respectively.
解法：在每一行做cols个选择，是否放在这一列。
利用一个colplacement list，每个位置代表当前那一行所选取的列.
'''

def solveNQueens(self, n):
    # write your code here
    
    if not n:
        return []
    res = []
    colPlacement = []
    self.helper(n, colPlacement, 0, res)

    return res

def helper(self, n, colPlacement, row, res):
    if len(colPlacement) == n:
        rows = self.draw(colPlacement)
        res.append(rows)
        return 
    
    for col in range(n):
        colPlacement.append(col)
        if self.isValid(colPlacement):
            self.helper(n, colPlacement, row + 1, res)
        colPlacement.pop()
    

def isValid(self, colPlacement):
    row = len(colPlacement) - 1
    col = colPlacement[row]
    
    for i in range(len(colPlacement) - 1):
        j = colPlacement[i]
        if j == col:
            return False
        if abs(j - col) / abs(i - row) == 1:
            return False
        
    return True
        
def draw(self, colPlacement):
    board = []
    for i in range(len(colPlacement)):
        colNum = colPlacement[i]
        row = ""
        for j in range(len(colPlacement)):
            if colNum != j:
                row += "."
            else:
                row += "Q"
        board.append(row)
    return board

'''
52. Next Permutation
Given a list of integers, which denote a permutation.
Find the next permutation in ascending order.
解法： 找到下一个排列
首先从后向前找到一个不是递增的位置i，把这个位置与从后往前第一个比它大的数交换位置。
然后对i+1 到 n-1 排序。（因为是从后向前递增，所以双指针不断交换首尾位置也能得到i+1后最小的排列）
'''
def nextPermutation(self, nums):
    # write your code here
    if not nums:
        return []
        
    n = len(nums)
    
    i = n - 1
    while i > 0 and nums[i] <= nums[i - 1]:
        i -= 1
    
    if i != 0:
        j = n -1
        while j > i - 1 and nums[j] <= nums[i - 1]:
            j -= 1
    
        nums[i - 1], nums[j] = nums[j], nums[i - 1]
    
    nums[i:] = sorted(nums[i:])
    
    return nums

'''
有重复元素：

'''
def nextPermutation(self, nums):
    # write your code here
    if not nums:
        return None
    n = len(nums)
    if n == 1:
        return nums
    
    i = n - 1
    
    while i > 0 and nums[i] <= nums[i - 1]:
        i -= 1
        
    j = n - 1
    if i > 0:
        while j > i and nums[j] <= nums[i - 1]:
            j -= 1
    
    nums[i - 1], nums[j] = nums[j], nums[i - 1]
    
    
    nums[i:] = nums[i:][::-1]    
    return nums

'''
197. Permutation Index
Given a permutation which contains no repeated number, 
find its index in all the permutations of these numbers, 
which are ordered in lexicographical order. The index begins at 1.
从后向前，选取坐标位置，看右边有多少数字比坐标数字小 smaller， 用smaller * 坐标右边位置的阶乘
通过从后向前的方法，可以记录下阶乘 fact *= n - i
'''
def permutationIndex(self, A):
    # write your code here
    
    if not A:
        return 1
    
    n = len(A)
    
    fact = 1
    perIndex = 0
    
    for i in range(n - 1, -1, -1):
        pivot = A[i]
        smaller = 0
        for j in range(n - 1, i, -1):
            if A[j] < pivot:
                smaller += 1
        perIndex += smaller * fact
        fact *= n - i    
    
    return perIndex + 1


'''
含有重复元素的permutation Index
利用hashmap来记录当前位置的重复元素数量
'''
def permutationIndexII(self, A):
    # write your code here
    
    if not A:
        return 1
    n = len(A)
    index = 0
    fact = 1
    repeat = {}
    repeatFact = 1
    
    for i in range(n - 1, -1, -1):
        if A[i] in repeat:
            repeat[A[i]] += 1
        else:
            repeat[A[i]] = 1
            
        repeatFact *= repeat[A[i]]
        
        smaller = 0
        for j in range(i + 1, n):
            if A[j] < A[i]:
                smaller += 1
        
        index += smaller * fact // repeatFact
        fact *= (n - i)
        
    return index + 1

'''
425. Letter Combinations of a Phone Number

Given a digit string excluded 0 and 1, return all possible letter 
combinations that the number could represent.
注意点： 其实就是长度为len(digits)的全组合
错误点： 因为是从digits中往下选，startindex代表当前的digit，
在mapping[digits[startIndex]]中选择一个char加入subset
然后再在下一个数字对应的字母中查找。
'''

def letterCombinations(self, digits):
    # write your code here
    
    if not digits:
        return []
    
    res = []
    self.helper(digits, 0, [], res)
    
    return res

def helper(self, digits, startIndex, subset, res):
    mapping = {
                "2" : "ABC",
                "3" : "DEF",
                "4" : "GHI",
                "5" : "JKL",
                "6" : "MNO",
                "7" : "PQRS",
                "8" : "TUV",
                "9" : "WXYZ"
                }
                
    if len(subset) == len(digits):
        res.append("".join(subset))
        return 
    
    for ch in mapping[digits[startIndex]]:
        subset.append(ch.lower())
        self.helper(digits, startIndex + 1, subset, res)
        subset.pop()

<<<<<<< HEAD
'''
570. Find the Missing Number II
Giving a string with number from 1-n in random order, 
but miss 1 number.Find that number.
注意点： 当取到0时，直接return
'''       

def findMissing2(self, n, str):
    # write your code here
    if not n:
        return None
    res = []
    
    self.helper(n, 0, [], res, str)
    print(res)
    for num in range(1, n + 1):
        if num not in res[0]:
            return num
        
def helper(self, n, start, path, res, str):
    if start == len(str):
        res.append(path[:])
    
    for i in range(1, 3):
        if start + i > len(str):
            return
        prefix = int(str[start : start + i])
        if prefix == 0:
            return 
        
        if prefix >= 1 and prefix <= n and prefix not in path:
            path.append(prefix)
            self.helper(n, start + i, path, res, str)
            path.pop()

'''
652. Factorization
A non-negative numbers can be regarded as product of its factors.
Write a function that takes an integer n and 
return all possible combinations of its factors.
'''
def getFactors(self, n):
    # write your code here

    start = 2
    res = []
    self.helper(n, start, [], res)
    
    return res



def helper(self, n, start, combination, res):
    if len(combination) != 0:
        res.append(combination[:] + [n])
        
    
    for i in range(start, int(math.sqrt(n)) + 1):
        if n % i != 0:
            continue

        combination.append(i)
        self.helper(n // i, i, combination, res)
        combination.pop()


=======


'''
house robberII
做两遍dp，从第一个到倒数第二个，从第二个到最后一个.
'''

def rob(self, nums: List[int]) -> int:
    if not nums:
        return 0
    n = len(nums)
    if n <= 2:
        return max(nums)
    dp = [0 for _ in range(n)]

    
    dp[0] = nums[0]
    dp[1] = max(dp[0], nums[1])
    
    for i in range(2, n - 1):
        dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
    
    mostAmount = dp[n - 2]
    
    dp[1] = nums[1]
    dp[2] = max(dp[1], nums[2])
    for i in range(3, n):
        dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
    
    return max(dp[n - 1], mostAmount)

'''
535. House Robber III
The thief has found himself a new place for his thievery again. There is only one entrance to this area, called the "root."
 Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that 
 "all houses in this place forms a binary tree". It will automatically contact the police if two directly-linked houses 
 were broken into on the same night.
解法： 使用递归，每个点有两个状态，抢或者不抢。 抢就是当前root.val + leftNoRoot + rightNoRoot
不抢就是 max(left.noRoot, leftRoot) + max(rightRoot + rightNoRoot)
'''
def houseRobber3(self, root):
    # write your code here
    if not root:
        return 0
    
    roob, noRoob = self.helper(root)
    return max(roob, noRoob)
    
    
def helper(self, root):
    if not root:
        return 0, 0
    
    left = self.helper(root.left)
    right = self.helper(root.right)
    
    roob = root.val + left[0] + right[0]
    noRoob = max(left[0], left[1]) + max(right[0], right[1])
    
    return noRoob, roob
>>>>>>> dc2ff96d8cba96c5dd5030142e3077372173b270
