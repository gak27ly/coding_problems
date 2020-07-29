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
出错处：在helper中使用了index导致出错，应该使用i
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
中文English
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




