#Inseration sort O(n^2)

def inserationSort(self, A):
	n = len(A)
	for i in range(1, n):
		key = A[i]

		j = i - 1
		#发现key比前面的数字小就把前面的数字往后移动一次
		while j >= 0 and A[j] > key:
			A[j + 1] = A[j]
			j -= 1
		A[j + 1] = key

#merge sort 
#将array分成左右两边，利用分治思想把两个sort好的左右array 合并
#合并的时候使用i，j来标记左右array当前需要被加入原array的index，将两个位置中较小的数字加入array update i，j 和 index 位置
def mergeSort(A):
	n = len(A)
	if n > 1:
		mid = n //2
		left = A[:mid]
		right = A[mid:]

		mergeSort(left)
		mergeSort(right)
		merge(A, left, right)


def merge(myList, left, right):
      # Two iterators for traversing the two halves
    i = 0
    j = 0
    # Iterator for the main list
    k = 0
    
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
          # The value from the left half has been used
          myList[k] = left[i]
          # Move the iterator forward
          i += 1
        else:
            myList[k] = right[j]
            j += 1
        # Move to the next slot
        k += 1

    # For all the remaining values
    while i < len(left):
        myList[k] = left[i]
        i += 1
        k += 1

    while j < len(right):
        myList[k]=right[j]
        j += 1
        k += 1

'''
partition method：让以后一个数字作为pivot，使用 k = start 作为小于pivot的数的位置坐标
然后在当前数组 i in range(start, end) 把所有小于pivot的数字交换到位置k。最后把pivot交换到位置k
这样 start 到 k-1 都小于pivot， k+1到end都大于pivot
利用recursion来sort start - k-1， k+1 - end
'''
def quickSort(A):
	quickSort1(A, 0, len(A) - 1)


def quickSort1(A, start, end):
	pivot = A[end]
	if start < end:
		pi = partition(A, start, end)
		quickSort1(A, start, pi - 1)
		quickSort1(A, pi + 1, end)

def partition(A, start, end):
	pivot = A[end]
	k = start

	for i in range(start, end):
		if A[i] < pivot:
			A[i], A[k] = A[k], A[i]
			k += 1

	A[k], A[end] = A[end], A[k]	
	return k


'''
selection sort
把起始点作为最小值，从之后的数字中更新最小值然后把最小值放到起始点
移动起始点到下一个位置
'''
def selectionSort(A):
	for i in range(len(A)):
		minIndex = i
		j = i + 1
		while j <= len(A) - 1:
			if A[j] < A[minIndex]:
				minIndex = j
			j += 1

		A[i], A[minIndex] = A[minIndex], A[i]


'''
Bubble Sort
从头开始，把大的数字挨个向后移动
'''

def bubbleSort(A):
	for i in range(len(A)):
		for j in range(len(A) - i - 1):
			if A[j] > A[j+1]:
				A[j], A[j+1] = A[j+1], A[j]
	








 