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


        




