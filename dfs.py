

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
A message containing letters from A-Z is being encoded to numbers using the following mapping:

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
135. Combination Sum
Given a set of candidtate numbers candidates and a target number target. 
Find all unique combinations in candidates where the numbers sums to target.
The same repeated number may be chosen from candidates unlimited number of times.
注意点： 去重后，使用startIndex来防止再次传入之前的数字
'''
def combinationSum(self, candidates, target):
    # write your code here
    
    if not candidates:
        return [[]]
    newcandidates = sorted(list(set(candidates)))
    
    res = [] 
    self.helper(newcandidates, 0, target, [], res)
    return res

def helper(self, candidates, startIndex, target, combination, res):
    if target == 0:
        res.append(list(combination))
        return
    if target < 0:
        return 
    
    for i in range(startIndex, len(candidates)):
        combination.append(candidates[i])
        self.helper(candidates, i, target - candidates[i], combination, res)
        combination.pop()
'''
153. Combination Sum II
中文English
Given an array num and a number target. 
Find all unique combinations in num where the numbers sum to target.

Each number in num can only be used once in one combination.
All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
注意点： 不能重复使用统一数字，需要从i + 1开始找
'''
def combinationSum2(self, nums, target):
    # write your code here
    if not nums:
        return []
    newNums = sorted(nums)
    res = []
    
    self.helper(newNums, 0, target, [], res)
    return res

def helper(self, nums, startIndex, target, combination, res):
    if target == 0:
        res.append(list(combination))
    
    if target < 0:
        return
    lastItem = None
    
    for i in range(startIndex, len(nums)):
        if nums[i] > target:
            continue
        if lastItem == nums[i]:
            continue
        combination.append(nums[i])
        self.helper(nums, i + 1, target - nums[i], combination, res)
        lastItem = combination.pop()
'''
10. String Permutation II
Given a string, find all permutations of it without duplicates.
注意点： 字符串需要先排序，否则无法去重
Example

Input: "aabb"
Output:
["aabb", "abab", "baba", "bbaa", "abba", "baab"]
'''
def stringPermutation2(self, str):
    # write your code here
    if not str:
        return [""]
        
    visited = [False for _ in range(len(str))]
    
    str =sorted(list(str))
    
    res = []
    
    self.helper(str, visited, [], res)
    
    return res

def helper(self, str, visited, premutation, res):
    if len(premutation) == len(str):
        res.append("".join(premutation))
        return
    
    lastChar = None
    
    for i in range(len(str)):
        if visited[i]:
            continue
        
        if lastChar == str[i]:
            continue
        
        premutation.append(str[i])
        visited[i] = True
        self.helper(str, visited, premutation, res)
        lastChar = premutation.pop()
        visited[i] = False  