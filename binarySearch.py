def findPosition(self, A, target):
    # Write your code here
    if len(A) == 0 or A == None:
        return -1
    
    start = 0
    end = len(A) - 1
    
    if target < A[start] or target > A[end]:
        return -1
    
    while start + 1 < end:
        mid = start + (end - start) / 2
        if target == A[mid]:
            return mid
        elif target > A[mid]:
            start = mid
        else:
            end = mid
    
    if target == A[end]:
        return end
    elif target == A[start]:
        return start
    else:
        return -1

def lastPosition(self, nums, target):
    # write your code here
    if not nums:
        return -1
        
    start, end = 0, len(nums) - 1
    
    
    while start + 1 < end:
        mid = (start + end) // 2
        pivot = nums[mid]
        #需要把大于target的数字都移出范围
        if pivot > target:
            end = mid
        else:
            #把小于target的数字移出，等于target的数字留下
            start = mid
    
    if nums[end] == target:
        return end
    if nums[start] == target:
        return start
    
    return -1

class Solution:
    """
    @param A: an integer array
    @param target: An integer
    @param k: An integer
    @return: an integer array
    """
    def kClosestNumbers(self, A, target, k):
        # write your code here
        
        index = self.findIndex(A, target)
        res = []

        left, right = index - 1, index
        for _ in range(k):
            if self.leftClose(left, right, target, A):
                res.append(A[left])
                left -= 1
            else:
                res.append(A[right])
                right += 1

        return res

'''
Find k closest number
解法：找到第一个大于等于target的数子位置index，从index -1 和index向左向右找k个数字
'''   

def findIndex(self, A, target):
    start, end = 0, len(A) - 1

    while start + 1 < end:
        mid = start + (end - start) // 2

        if A[mid] >= target:
            end = mid
        else:
            start = mid
    
    if A[start] >= target:
        return start
    if A[end] >= target:
        return end
        
    return len(A)
    
def leftClose(self, left, right, target, A):
    if left < 0:
        return False
    if right > len(A) - 1:
        return True
    
    if left >= 0 and right < len(A):
        if target - A[left] <= A[right] - target:
            return True
    
    return False

'''
find mountain peak
解法：只要对面前面一个index就能知道当前是否是在升序中
'''
def peakIndexInMountainArray(self, A):
    # Write your code here
    start, end = 0, len(A)
    
    while start + 1 < end:
        mid = (start + end) // 2
        
        if A[mid] > A[mid - 1]:
            start = mid
        elif A[mid] < A[mid - 1]:
            end = mid
        
    if A[start] > A[end]:
        return start
    else:
        return end
'''
Suppose a sorted array in ascending order is rotated at 
some pivot unknown to you beforehand.
No duplicate exists
解法： 找到中间位置和结尾比较大小 
'''      

def findMin(self, nums):
    # write your code here
    if not nums:
        return None
    start, end = 0, len(nums) - 1
    
    while start + 1 < end:
        mid = (start + end) // 2
        
        if nums[mid] > nums[end]:
            start = mid
        else:
            end = mid
        
    if nums[start] < nums[end]:
        return nums[start]
    return nums[end]

class Solution:
    """
    @param A: an integer rotated sorted array
    @param target: an integer to be searched
    @return: an integer
    解法： 注重有序的那部分
    注意：需要把等于的部分考虑进去，否则会出错
    """
def search(self, A, target):
    # write your code here
    if not A:
        return -1
    start, end = 0, len(A) - 1
    
    while start + 1 < end:
        mid = (start + end) // 2
        
        if A[mid] < A[end]:
            if A[mid] <= target <= A[end]:
                start = mid
            else:
                end = mid
        else:
            if A[start] <= target <= A[mid]:
                end = mid
            else:
                start = mid
    
    if A[start] == target:
        return start
    if A[end] == target:
        return end
    
    return -1

'''
            
There is an integer array which has the following features:

The numbers in adjacent positions are different.
A[0] < A[1] && A[A.length - 2] > A[A.length - 1].
We define a position P is a peak if:

A[P] > A[P-1] && A[P] > A[P+1]
Find a peak element in this array. Return the index of the peak.

It's guaranteed the array has at least one peak.
The array may contain multiple peeks, find any of them.
The array has at least 3 numbers in it.
解法： 找到中间位置 和左右相比，哪边大就保留哪边，如果中间大那就是peak.
'''
def findPeak(self, A):
# write your code here
    start, end = 1, len(A) - 2
    while start + 1 <  end:
        mid = (start + end) // 2
        if A[mid] < A[mid - 1]:
            end = mid
        elif A[mid] < A[mid + 1]:
            start = mid
        else:
            return mid

    if A[start] < A[end]:
        return end
    else:
        return start
            

