'''
从m位置到n位置reverse一个linked list
'''

def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
    
    cur = head
    count = 0
    dummy = ListNode(0)
    dummy.next = head 
    pre = dummy
    
    for _ in range(m - 1):
        pre = pre.next
        
    cur = pre.next
    
    for _ in range(n - m):
        next = cur.next
        cur.next = next.next
        next.next = pre.next
        pre.next = next
    
    return dummy.next

'''
merge K sorted List
使用merge sort的方法对每两个list进行merge
'''


def mergeKLists(self, lists):
    if not lists:
        return None
    
    return self.merge_range_lists(lists, 0, len(lists) - 1)
    
def merge_range_lists(self, lists, start, end):
    if start == end:
        return lists[start]
    
    mid = (start + end) // 2
    left = self.merge_range_lists(lists, start, mid)
    right = self.merge_range_lists(lists, mid + 1, end)
    return self.merge_two_lists(left, right)
    
def merge_two_lists(self, head1, head2):
    tail = dummy = ListNode(0)
    while head1 and head2:
        if head1.val < head2.val:
            tail.next = head1
            head1 = head1.next
        else:
            tail.next = head2
            head2 = head2.next
        tail = tail.next
        
    if head1:
        tail.next = head1
    if head2:
        tail.next = head2
    
    return dummy.next
'''
使用pority queue 实现
先把每个list heappush进入[]
然后pop出最小的head，再把head.next push进去[]
'''

def mergeKLists(self, lists):
    if not lists:
        return None
    
    dummy = ListNode(0)
    tail = dummy
    heap = []
    for head in lists:
        if head:
            heapq.heappush(heap, head)
            
    while heap:
        head = heapq.heappop(heap)
        tail.next = head
        tail = head
        if head.next:
            heapq.heappush(heap, head.next)
                
    return dummy.next
