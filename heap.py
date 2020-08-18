"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
注意点： heapq 只能保证最小的值被pop出，所以把要排序的值取负再放入来
去除太远的点。存入坐标时取负值保证弹出的顺序。
"""
import heapq
class Solution:
    """
    @param points: a list of points
    @param origin: a point
    @param k: An integer
    @return: the k closest points
    """
    def kClosest(self, points, origin, k):
        # write your code here
        self.heap = []
        res = []
        
        for point in points:
            distance = self.compare(point, origin)
            heapq.heappush(self.heap, (-distance, -point.x, -point.y))
            if len(self.heap) > k:
                heapq.heappop(self.heap)
            
            
        
        while self.heap:
            _, x, y = heapq.heappop(self.heap)
            
            res.append(Point(-x, -y))
        
        res.reverse()
        
        return res


import heapq

class Solution:
    """
    @param nums: an integer array
    @param k: An integer
    @return: the top k largest numbers in array
    """
    def topk(self, nums, k):
        # write your code here
        for i in range(len(nums)):
            nums[i] = -nums[i]
            
        heapq.heapify(nums)
        res = []
        while k:
            res.append(-heapq.heappop(nums))
            k -= 1
        return res
    
    def compare(self, a, b):
        return (b.x - a.x) ** 2 + (b.y - a.y) ** 2
'''
401. Kth Smallest Number in Sorted Matrix
Find the kth smallest number in a row and column sorted matrix.
Each row and each column of the matrix is incremental.
解法： 先把每一行的第一个数字和该数字的行数列数传入heap.
每pop出来一个数字就push该数字的下一个 知道pop出来了k个数字为止
'''
def kthSmallest(self, matrix, k):
    # write your code here
    heap = []
    n = len(matrix)
    if n == 1:
        return matrix[0][k - 1]
        
    for i in range(len(matrix)):
        heapq.heappush(heap, (matrix[i][0], i, 0))
    
    while k:
        item = heapq.heappop(heap)
        k -= 1
        row, index  = item[1], item[2]
        if index == len(matrix[row]) - 1:
            continue
        
        heapq.heappush(heap, (matrix[row][index + 1], row, index + 1))
    
    return item[0]




class LinkedNode:
    
    def __init__(self, key=None, value=None, next=None):
        self.key = key
        self.value = value
        self.next = next

class LinkedNode:
    
    def __init__(self, key=None, value=None, next=None):
        self.key = key
        self.value = value
        self.next = next
        
class LRUCache:

    # @param capacity, an integer
    def __init__(self, capacity):
        self.key_to_prev = {}
        self.dummy = LinkedNode()
        self.tail = self.dummy
        self.capacity = capacity
    
    def push_back(self, node):
        self.key_to_prev[node.key] = self.tail
        self.tail.next = node
        self.tail = node
    
    def pop_front(self):
        # 删除头部
        head = self.dummy.next
        del self.key_to_prev[head.key]
        self.dummy.next = head.next
        self.key_to_prev[head.next.key] = self.dummy
        
    # change "prev->node->next...->tail"
    # to "prev->next->...->tail->node"
    def kick(self, prev):   #将数据移动至尾部
        node = prev.next
        if node == self.tail:
            return
        
        # remove the current node from linked list
        prev.next = node.next
        # update the previous node in hash map
        self.key_to_prev[node.next.key] = prev
        node.next = None

        self.push_back(node)

    # @return an integer
    def get(self, key):
        if key not in self.key_to_prev:
            return -1
        
        prev = self.key_to_prev[key]
        current = prev.next
        
        self.kick(prev)
        return current.value

    # @param key, an integer
    # @param value, an integer
    # @return nothing
    def set(self, key, value):
        if key in self.key_to_prev:    
            self.kick(self.key_to_prev[key])
            self.key_to_prev[key].next.value = value
            return
        
        self.push_back(LinkedNode(key, value))  #如果key不存在，则存入新节点
        if len(self.key_to_prev) > self.capacity:       #如果缓存超出上限
            self.pop_front()                    #删除头部