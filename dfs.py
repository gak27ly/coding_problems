

def longestIncreasingPath(matrix):
    if not matrix:
        return 0
    n = len(matrix)
    m = len(matrix[0])
    ans = 0
    for i in range(n):
        for j in range(m):
            ans = max(helper(matrix, i, j, []), ans)
    return ans
    
def helper(matrix, x, y, visited):
    steps = 1
    DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if (x, y) not in visited:
        visited.append((x, y))
    
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if isValid(matrix, nx, ny) and (nx, ny) not in visited and \
            matrix[nx][ny] - matrix[x][y] > 3:
            steps = max(helper(matrix, nx, ny, visited) + 1, steps)
            
    visited.pop()
    
    return steps

def isValid(matrix, x, y):
    if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]):
        return True
    return False
    
    
A = [[1, 2, 3], [4, 5, 6],[7, 8, 9]]
print(longestIncreasingPath(A))




'''
512. Decode Ways
中文English
A message containing letters from A-Z is being encoded to numbers using the following mapping:

'A' -> 1
'B' -> 2

'''

def numDecodings(s):

    if not s or len(s) == 0:
        return []
        
    n = len(s)
    
    res = []
    
    dfs(s, [], res)
    res = sorted(res, key = len)
    return res
        
def dfs(s, decode, res):
    if len(s) == 0:
        res.append("".join(decode))
        return 
    
    if s[0] == "0":
        return

    for i in range(1, len(s) + 1):
        prefix = s[:i]
        if prefix[0] == "0":
            continue
        if int(prefix) > 26:
            break
        
        decode.append(chr(int(prefix) + 96))
        dfs(s[i:], decode, res)
        decode.pop()
            
print(numDecodings("105"))