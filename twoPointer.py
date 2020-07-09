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
解法： 选定指针，直接三个数值相加打擂台.记录下最小差值的total
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
        	#为了使写入次数最少，想等情况下无需写入
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
