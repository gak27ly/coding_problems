def roatedArray(self, A, target):
  start, end = 0, len(A) - 1


  while start + 1 < end:
    mid = (start + end) // 2
    if A[mid] < target and target < A[end]:
      start = mid
   	elif A[mid] < target and target > A[end]:
      end = mid
    elif A[mid] > target and target < A[start]:
      start = mid
    elif A[mid] > target and target > A[start]:
      end = mid
      
  if A[start] == target:
    return start
  if A[end] == target:
    return end

  return -1


  先找到roated point， 再用 target 和 头去比


  