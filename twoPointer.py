'''
相向双指针
1. Reverse类
2. Two Sum类（最多）
3. Partition类 （较多）
'''


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
