# •初始时，把待排序序列中的n个记录看成n个有序子序列（因为一个记录的序列总是排好序的），每个子序列的长度均为 1。
# •把当时序列组里的有序子序列两两归并，完成一遍后序列组里的排序序列个数减半，每个子序列的长度加倍。
# 。对加长的有序子序列重复上面的操作，最终得到一个长度为n的有序序列
#第一步：申请空间，使其大小为两个已经排序序列之和，该空间用来存放合并后的序列
# 第二步：设定两个指针，最初位置分别为两个已经排序序列的起始位置
# 第三步：比较两个指针所指向的元素，选择相对小的元素放入到合并空间，并移动指针到下一位置

# 归并排序
def merge_sort(num_list):
    length = len(num_list)

    # 递归终止退出条件
    if length <= 1:
        return num_list

    # 拆分
    mid = length // 2
    left_l = merge_sort(num_list[:mid])   # 对左侧的列表进行排序
    right_l = merge_sort(num_list[mid:])  # 对右侧的列表进行排序

    # merge 合并操作
    # 初始化两个指针p, q 初始位置为起始位置，初始化一个临时数组temp_list
    p, q, temp_list = 0, 0, list()
    len_left, len_right = len(left_l), len(right_l)  # 计算当前被合并的列表的长度

    while len_left > p and len_right > q:
        if left_l[p] <= right_l[q]:
            temp_list.append(left_l[p])
            p += 1
        else:
            temp_list.append(right_l[q])
            q += 1
    # 如果left 和 right 的长度不相等，把长的部分直接追加到列表中
    temp_list += left_l[p:]
    temp_list += right_l[q:]

    return temp_list


if __name__ == '__main__':
    num_list = [44, 23, 1, 14, 6, 9, 4, 5, 33]
    new_list = merge_sort(num_list)
    for k, v in enumerate(new_list):
        num_list[k] = v
    print('num_list:', num_list)

