'''
相向双指针
1. Reverse类
2. Two Sum类（最多）
3. Partition类 （较多）
'''

#不对string进行去重， 如遇到非alnum的操作，直接在遇到的时候跳过即可
def isPalindrome(self, s):
    s = s.lower()
    start, end = 0, len(s) - 1 
    while start < end:
        if not s[start].isalnum():
            start += 1 
            continue
        if not s[end].isalnum():
            end -= 1 
            continue 
        if s[start] != s[end]:
            return False  
        start += 1 
        end -= 1 
    return True

"""
@param s: a string
@return bool: whether you can make s a palindrome by deleting at most one character
解法： 想找到不相同的位置，去掉左边看右边是否是valid，或去掉右边看左边是否是valid
"""
def validPalindrome(self, s):
    # Write your code here
    l, r = 0, len(s) - 1
    while l < r:
        if s[l] != s[r]:
            break
        l += 1
        r -= 1
    
    if l >= r:
        return True
    
    return self.isValidPalindorme(s, l + 1, r) or self.isValidPalindorme(s, l, r - 1)
    

def isValidPalindorme(self, s, start, end):
    while start < end:
        if s[start] != s[end]:
            break
        start += 1
        end -= 1
    if start >= end:
        return True
    
    return False
'''
解法1: 放入dict 加入时候计数 查询的时候看是否dict中有对应的数字
注意如果target和对应数字数量小于2 则不能return True
'''
class TwoSum:
    """
    @param number: An integer
    @return: nothing
    """
    mapping = {}
    
    def add(self, number):
        # write your code here
        if number in self.mapping:
            self.mapping[number] += 1
        else:
            self.mapping[number] = 1
    """
    @param value: An integer
    @return: Find if there exists any pair of numbers which sum is equal to the value.
    """
    def find(self, value):
        # write your code here
        for val in  self.mapping:
            target = value - val
            if target in self.mapping:
                if target == val and self.mapping[target] <= 1:
                    continue
                return True
        return False
'''
解法2: 加入的数字全部加入[], sort以后再利用两个指针从首位开始扫描
对于已经sort过的数组可以直接扫描
'''
class TwoSum:
	arr = []
	def add(self, number):
	    # write your code here
	    self.arr.append(number)

	def find(self, value):
	    # write your code here
	    l, r = 0, len(self.arr) - 1
	    self.arr.sort()
	    
	    while l < r:
	        total = self.arr[l] + self.arr[r]
	        if total == value:
	            return True
	        if total < value:
	            l += 1
	        else:
	            r -= 1
	    return False

'''
不能有相同的pair 
解法： 当发现现在的和为target时候，移动两个指针直到和上个位置不相等
'''
def twoSum6(self, nums, target):
    if not nums or len(nums) < 2:
        return 0

    nums.sort()
    
    count = 0
    left, right = 0, len(nums) - 1
    
    while left < right:
        if nums[left] + nums[right] == target:
            count, left, right = count + 1, left + 1, right - 1
            while left < right and nums[left] == nums[left - 1]:
                left += 1
            while left < right and nums[right] == nums[right + 1]:
                right -= 1
        elif nums[left] + nums[right] > target:
            right -= 1
        else:
            left += 1
    return count

'''
609. Two Sum - Less than or equal to target
Given an array of integers, find how many pairs in the array such that their 
sum is less than or equal to a specific target number. Please return the number of pairs.
解法：如果发现 <= target就把end - start都加起来 再移动start。如果>target则移动end
'''
def twoSum5(self, nums, target):
    # write your code here
    if not nums or len(nums) < 2:
        return 0
    nums.sort()
    res = 0
    start, end = 0, len(nums) - 1
    
    while start < end:
        if nums[start] + nums[end] <= target:
            res += end - start
            start += 1
        else:
            end -= 1
    return res

'''
Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? 
Find all unique triplets in the array which gives the sum of zero.
解法：简化为2sum题。 先确定一个定点以它的负值作为target，再从之后的数组中找到twosum的target
'''

def threeSum(self, numbers):
    # write your code here
    if not numbers:
        return []
    
    numbers.sort()
    n = len(numbers)
    target = 0
    res = []
    
    for i in range(len(numbers) - 2):
        target = -numbers[i]
        if i > 0 and numbers[i] == numbers[i - 1]:
            continue
        
        self.twoSum(numbers, i + 1, target,res)

    return res
        
def twoSum(self, numbers, startIndex, target,res):
    l, r = startIndex, len(numbers) - 1

    while l < r:
        if numbers[l] + numbers[r] > target:
            r -= 1
        elif numbers[l] + numbers[r] < target:
            l += 1
        else:
            res.append([-target, numbers[l], numbers[r]])
            r -= 1
            l += 1
            while l < r and numbers[l] == numbers[l - 1]:
                l += 1
            while l < r and numbers[r] == numbers[r + 1]:
                r -= 1

'''
382. Triangle Count
Given an array of integers, how many three numbers can be found in the array, 
so that we can build an triangle whose three edges length is the three numbers that we find?
Example 1:

Input: [3, 4, 6, 7]
Output: 3
Explanation:
They are (3, 4, 6), 
         (3, 6, 7),
         (4, 6, 7)
Example 2:
Input: [4, 4, 4, 4]
Output: 4
'''
def triangleCount(self, S):
    # write your code here
    if not S or len(S) < 3:
        return 0
    
    S.sort()
    res = 0
    for i in range(len(S) - 1, 1, -1):
        start, end = 0, i - 1
        while start < end:
            if S[start] + S[end] > S[i]: #当满足a+b>c时，a-b之间的值都能作为第一个数
                res += end - start
                end -= 1
            else:
                start += 1
    return res
'''
Three Sum Closest
解法：确定一个点以后从此点开始为start，len(nums) - 1为end
来进行双指针算法
'''

def threeSumClosest(self, numbers, target):
    # write your code here
    
    if not numbers or len(numbers) < 3:
        return 0
        
    numbers.sort()
    totoal, res = 0, sys.maxsize

    for i in range(len(numbers) - 2):
        start, end = i + 1, len(numbers) - 1
        
        while start < end:
            total = numbers[start] + numbers[end] + numbers[i]
            if abs(total - target) < abs(res - target):
                res = total
            if total > target:
                end -= 1
            elif total < target:
                start += 1
            else:
                return target
    return res

'''
Find the kth smallest number in an unsorted integer array.
解法： quickSelect
'''
def kthSmallest(self, k, nums):
    # write your code here
    return self.quickSelect(k - 1, nums, 0, len(nums) - 1)
    
def quickSelect(self, k, nums, start, end):
    if start == end:
        return nums[start]
    pivot = nums[end]
    
    left, right = start, end
    
    
    while start <= end:
        while start <= end and nums[start] < pivot:
            start += 1
        while start <= end and nums[end] > pivot:
            end -= 1
            
        if start <= end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    
    if left <= k <= end:
        return self.quickSelect(k, nums, left, end)
    elif start <= k <= right:
        return self.quickSelect(k, nums, start, right)
    else:
        return nums[k]

'''
同向双指针
'''

def middleNode(self, head):
    if not head:
        return None
    left = right = head
    
    while right.next:
        steps = 2
        while steps > 0 and right.next:
            right = right.next
            steps -= 1
        if steps == 0:
            left = left.next
    return left
'''
解法： l指针作为当前下一个非0位置
r指针遍历，当r找到非零数字，把数字丢给l
'''
def moveZeroes(self, nums):
    # write your code here
    l, r = 0, 0
    n = len(nums)
    
    while r < n:
        if nums[r] != 0:
        	#为了使写入次数最少，相等情况下无需写入
            if nums[r] != nums[l]:
                nums[l] = nums[r]
            l += 1
        r += 1
        
    while l < n:
    	#为了减少写入，如果本来就是0则无需写入
        if nums[l] != 0:
            nums[l] = 0
        l += 1

'''
解法1：allow extra space
用set记录下已经遍历过的数字
将没有遇到过的数字甩给左指针，最后return 左指针位置 
'''

 def deduplication(self, nums):
    # write your code here
    mapping = set()
    l = r = 0
    n = len(nums)
    
    while r < n:
        if nums[r] not in mapping:
            mapping.add(nums[r])
            nums[l] = nums[r]
            l += 1
        r += 1
    
    return l
'''
解法2: sort以后遍历数组，发现与前面不同的数字就放到l指针的位置上，再移动l指针
'''
def deduplication(self, nums):
    # write your code here
    n = len(nums)
    if n == 0:
        return 0
        
    nums.sort()
    
    l = 1
    
    for i in range(1, n):
        if nums[i] != nums[i - 1]:
            nums[l] = nums[i]
            l += 1
            
    return l

'''
给定一个排序后的整数数组，找到两个数的 差 等于目标值。
你需要返回一个包含两个数字的列表[num1, num2], 使得num1与num2的差为target，同时num1必须小于num2。
解法： 从头开始的两个点求差值， 若大于target则做指针向右移动，这时需要看右指针位置来决定其是否移动
若target小于差值则移动右指针。 
'''
def twoSum7(self, nums, target):
    # write your code here
    if not nums or len(nums) < 2:
        return []
    target = abs(target) #注意需要问清楚差值是否可以是负数，如果可以要取target的绝对值
    i, j = 0, 1
    res = []
    while i < j and j < len(nums):
        if nums[j] - nums[i] < target:
            j += 1
        elif nums[j] - nums[i] > target:
            if j == i + 1:
                j += 1
            i += 1
        else:
        	return [nums[i],nums[j]]
 
    return [-1, -1]
'''
144. Interleaving Positive and Negative Numbers

Given an array with positive and negative integers. Re-range it to interleaving with positive and negative integers.

Example
Example 1

Input : [-1, -2, -3, 4, 5, 6]
Outout : [-1, 5, -2, 4, -3, 6]
解法：首先要知道正负数哪个多，多的那个放在第一个位置
然后通过同向双指针找到下一个需要交换的位置进行交换
'''
def rerange(self, A):
    # write your code here
    
    negative = 0
    for i in range(len(A)):
        if A[i] < 0:
            negative += 1
    if negative > len(A) - negative:
        neg, pos = 0, 1
    else:
        neg, pos = 1, 0
        
    n = len(A)
    
    while neg < n and pos < n:
        while neg < n and A[neg] < 0:
            neg += 2
        while pos < n and A[pos] > 0:
            pos += 2
        if neg < n and pos < n: 
            A[neg], A[pos] = A[pos], A[neg]
            neg += 2
            pos += 2