
/*
Sorting
*/

//MergeSort - topDown
// time nlogn space logn
public ListNode sortList(ListNode head) {
    if (head == null || head.next == null) return head;
    ListNode slow = head;
    ListNode fast = head.next;

    while(fast != null && fast.next != null) {
    	fast = fast.next.next;
    	slow = slow.next;
    }

    ListNode mid = slow.next;
    slow.next = null;
    return merge(sortList(mid), sortList(head));
}

// time Length of two list, space log n 
private ListNode merge(ListNode l1, ListNode l2) {
	ListNode dummy = new ListNode();
	ListNode tail = dummy;

	while(l1 !=null && l2 != null) {
		if (l1.val < l2.val) {
			tail.next = l1;
			tail = tail.next;
			l1 = l1.next;
		} else{
			tail.next = l2;
			tail = tail.next;
			l2 = l2.next;
		}
	}

	if (l1 != null) tail.next = l1;
	if (l2 != null) tail.next = l2;

	return dummy.next;
}

// bottom up
// 分成n组，从一个元素开始两两merge