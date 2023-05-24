# 一个序列中的记录没排好序，那么其中一定有逆序存在。如果交换所发现的逆序记录对，得到的序列将更接近排序序列；通过不断减少序列中的道序，最终可以得到排序序列。
# 1.每一遍检查可以把一个最大元素交换到位，些较大元素右移一段，可能移动很远。
# 2.从左到右比较，导致小元素一次只左移一位。个别距离目标位置很远的小元素
def bubble_sort(lst):
    for i in range(len(lst)):
        found = False
        for j in range(1,len(lst)-i):
            if lst[j-1] > lst[j]:
                lst[j-1],lst[j] = lst[j],lst[j-1]
                found = True
        if not found:
            break

    return lst

lst = [8,9,6,7,2,4,3]
print(bubble_sort(lst))