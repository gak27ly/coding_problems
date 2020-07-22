#prefixSumArr[i] = prefixSumArr[i-1] + arr[i-1]


'''
41. Maximum Subarray
Given an array of integers, find a contiguous subarray which has the largest sum.

'''
def maxSubArray(self, nums):
    # write your code here
    if not nums:
        return 0
        
    preFixSum = 0
    currentMaxSum = -sys.maxsize - 1
    minPreFixSum = 0
    
    
    for i in range(len(nums)):
        preFixSum += nums[i]
        currentMaxSum = max(preFixSum - minPreFixSum, currentMaxSum)
        minPreFixSum = min(preFixSum, minPreFixSum)
        
    return currentMaxSum


'''
42. Maximum Subarray II
中文English
Given an array of integers, find two non-overlapping subarrays which have the largest sum.
The number in each subarray should be contiguous.
Return the largest sum.
解法： 记录下两个list，分别是从左往右遍历时候的最大值和从右往左遍历时的最大值
找出对应的值相加找到最大结果
'''

def maxTwoSubArrays(self, nums):
    # write your code here
    prefixSumLeft = self.helper(nums)
    prefixSumRight = self.helper(nums[::-1])
    n = len(nums)
    
    res = -sys.maxsize - 1
    
    for i in range(n - 1):
        res = max(res, prefixSumLeft[i] + prefixSumRight[n - 2 - i])

    return res
    
def helper(self, nums):
    if not nums:
        return [0]
        
    n = len(nums)
    prefixSum = 0
    minPrefix = 0
    currentMax = -sys.maxsize - 1
    maxSum = [0] * n
    
    for i in range(n):
        prefixSum += nums[i]
        maxSum[i] = max(prefixSum - minPrefix, currentMax)
        currentMax = max(maxSum[i], currentMax)
        minPrefix = min(minPrefix, prefixSum)
        
    return maxSum

'''
665. Range Sum Query 2D - Immutable
Given a 2D matrix, find the sum of the elements inside the 
rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).
解法： 通过前缀和建立一个prefixSum 的array，用这个array的特点来求出区域范围的和
优化： 优化前缀和为一位数组需要将二位的prefixSum变为一维，在计算过程中求出upTotal 和 total，结果就是两者的差值
'''

def __init__(self, matrix):
    # do intialization if necessary
    n = len(matrix)
    m = len(matrix[0])
    self.prefixSum = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            self.prefixSum[i][j] = self.prefixSum[i - 1][j] + self.prefixSum[i][j - 1] - self.prefixSum[i - 1][j - 1] + matrix[i - 1][j - 1]
    
"""
@param: row1: An integer
@param: col1: An integer
@param: row2: An integer
@param: col2: An integer
@return: An integer
"""

def sumRegion(self, row1, col1, row2, col2):
    # write your code here
    res = 0
    row1 += 1
    col1 += 1
    row2 += 1
    col2 += 1
    
    res = self.prefixSum[row2][col2] - self.prefixSum[row1 - 1][col2] - self.prefixSum[row2][col1 - 1] + self.prefixSum[row1 - 1][col1 - 1]
    return res
