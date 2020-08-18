leetcode 
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