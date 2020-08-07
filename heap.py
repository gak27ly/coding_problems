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
    
    def compare(self, a, b):
        return (b.x - a.x) ** 2 + (b.y - a.y) ** 2