/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-10-04 14:27:37
 * @LastEditTime: 2019-10-04 15:12:23
 * @LastEditors: Please set LastEditors
 */
#include <vector>

using namespace std;
// 二分查找（折半查找）：对于已排序，若无序，需要先排序

// 非递归
int BinarySearch(vector<int> v, int value , int low, int high) {
	if (v.size() <= 0) {
		return -1;
	}
	while (low <= high) {
		int mid = low + (high - low) / 2;
		if (v[mid] == value) {
			return mid;
		}
		else if (v[mid] > value) {
			high = mid - 1;
		}
		else {
			low = mid + 1;
		}
	}

	return -1;
}

// 递归
int BinarySearch2(vector<int> v, int value, int low, int high)
{
	if (low > high)
		return -1;
	int mid = low + (high - low) / 2;
	if (v[mid] == value)
		return mid;
	else if (v[mid] > value)
		return BinarySearch2(v, value, low, mid - 1);
	else
		return BinarySearch2(v, value, mid + 1, high);
}
