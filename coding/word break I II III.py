# word break I  
#DFS
def wordBreak(self, s, dict):
    # write your code here

    return self.helper(s, dict, {})
    
    
def helper(self, s, dict, memo):
    if s in memo:
        return memo[s]
    
    if len(s) == 0:
        return True
    #边界为len(s) + 1 否则prefix只能取到倒数第二个字母
    for i in range(1, len(s) + 1):
        prefix = s[:i]
        
        if prefix not in dict:
            continue
        
        if self.helper(s[i:], dict, memo):
            memo[s] = True
            return True
        
    memo[s] = False
    return False

# DP
def wordBreak(self, s, dict):
    if not dict:
        return s == ""
    maxLength = max([len(word) for word in dict])

    dp = [False for _ in range(len(s) + 1)]
    dp[0] = True


    for i in range(1, len(s)+1):
        for j in range(max(i - maxLength, 0), i):
            if not dp[j]:
                continue
            if s[j:i] in dict:
                dp[i] = True
                break
        
    return dp[len(s)]

#word break II
# Given a string s and a dictionary of words dict, add spaces in s to construct a sentence where each word is a valid dictionary word.

# Return all such possible sentences.

# Input："lintcode"，["de","ding","co","code","lint"]
# Output：["lint code", "lint co de"]
def wordBreak(self, s, wordDict):
    # write your code here
    return self.helper(s, wordDict, {})
    
    
def helper(self, s, wordDict, memo):
    if s in memo:
        return memo[s]
        
    if len(s) == 0:
        return []
    
    result = []
    #先看是否是整个s都在字典里， 如果是的话要加入
    if s in wordDict:
        result.append(s)
    #拆分s， prefix只需要取到倒数第二个字母， 应为如果整个都在字典里会直接在上面加入result
    for i in range(1, len(s)):
        prefix = s[:i]
        if prefix not in wordDict:
            continue
        
        combinations = self.helper(s[i:], wordDict, memo)
        
        for string in combinations:
            result.append(prefix + " " + string)

    memo[s] = result
    
    return result



