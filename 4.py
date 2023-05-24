class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s : return 0
        left = 0
        lookup = set()
        max_len = 0
        que_len = 0
        for i in s:
            que_len+=1
            while i in lookup:
                lookup.remove(s[left])
                que_len -=1
                left +=1
            if que_len > max_len : max_len = que_len
            lookup.add(i)
        return max_len

a = "pwwkew"
k = Solution()
print(k.lengthOfLongestSubstring(a))
