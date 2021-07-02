def insert_sort(A):
	N = len(A)
	for top in range(1, N):
		k = top
		while k > 0 and A[k-1] > A[k]:
			A[k-1], A[k] = A[k], A[k-1]
			k -= 1

def choice_sort(A):
	N = len(A)
	for pos in range(0, N-1):
		for k in range(pos+1, N):
			if A[pos] > A[k]:
				A[pos], A[k] = A[k], A[pos]

def bubble_sort(A):
	N = len(A)
	for bypass in range(1, N):
		for k in range(0, N-bypass):
			if A[k] > A[k+1]:
				A[k], A[k+1] = A[k+1], A[k]

def count_sort(A):
	def max_num_in_list(A):
		number = 0
		for i in A:
			if i > number:
				number = i
		return number
	N = len(A)
	F = [0] * (max_num_in_list(A) + 1)
	for i in range(N):
		x = A[i]
		F[x] += 1
	A.clear()
	number = 0
	for i in F:
		amount = i
		while amount > 0:
			A.append(number)
			amount -= 1
		number += 1



def test_algorithm(tested_algorithm):
	print('Algorithm', tested_algorithm.__doc__)

	A = [3, 2, 5, 1, 4]
	B = [1, 2, 3, 4, 5]
	tested_algorithm(A)
	print('testcase#1:', ' OK' if A == B else ' FAIL')

	A = list(range(10, 20)) + list(range(0, 10))
	B = list(range(20))
	tested_algorithm(A)
	print('testcase#2:', ' OK' if A == B else ' FAIL')

	A = [3, 4, 5, 1, 4, 1]
	B = [1, 1, 3, 4, 4, 5]
	tested_algorithm(A)
	print('testcase#3:', ' OK' if A == B else ' FAIL')

# test_algorithm(insert_sort)
# test_algorithm(choice_sort)
# test_algorithm(bubble_sort)
# test_algorithm(count_sort)

def factorial(n):
	assert n >= 0, "incorrect input"
	if n == 0:
		return 1
	return factorial(n-1)*n

# for i in range(10):
	# print(f(i))

def fibonacci(n):
	assert n >= 0, "incorrect input"
	if n == 0:
		return 0
	elif n == 1:
		return 1
	else:
		return fibonacci(n-1)+fibonacci(n-2)

# for i in range(15):
# 	print(fibonacci(i))


def gcd(a, b):
	if b == 0:
		return a
	return gcd(b, a%b)

# print(gcd(51, 6))

def pow(a, n):
	if n == 0:
		return 1
	elif n%2 == 1: # нечетное
		return pow(a, n-1)*a
	else:
		return pow(a**2, n//2)

# print(pow(3, 7))



def generate_permutations(N, M=-1, prefix=None):
	def find(number, A):
		for i in A:
			if number == i:
				return True
		return False
	M = N if M == -1 else M
	prefix = prefix or []
	if M == 0:
		print(prefix)
		return
	for number in range(1, N+1):
		if find(number, prefix):
			continue
		prefix.append(number)
		generate_permutations(N, M-1, prefix)
		prefix.pop()

# generate_permutations(3)

def merge(A:list, B:list):
	C = [0] * (len(A) + len(B))
	i = k = n = 0
	while i<len(A) and k<len(B):
		if A[i] <= B[k]:
			C[n] = A[i]
			i += 1
			n += 1
		else:
			C[n] = B[k]
			k += 1
			n += 1
	while i < len(A):
		C[n] = A[i]
		i += 1
		n += 1
	while k < len(B):
		C[n] = B[k]
		i += 1
		n += 1
	return C

def merge_sort(A):
	if len(A) <= 1:
		return
	middle = len(A) // 2
	L = [A[i] for i in range(0, middle)]
	R = [A[i] for i in range(middle, len(A))]
	merge_sort(L)
	merge_sort(R)
	C = merge(L, R)

def hoar_sort(A):
	if len(A) <= 1:
		return 
	barrier = A[0]
	L = []
	M = []
	R = []
	for x in A:
		if x < barrier:
			L.append(x)
		elif x == barrier:
			M.append(x)
		else:
			R.append(x)
	hoar_sort(L)
	hoar_sort(R)
	k = 0
	for x in L + M + R:
		A[k] = x
		k += 1

def check_sorted(A, ascending=True):
	"""Проверка отсортированности за O(len(A))"""
	N = len(A)
	flag = True
	s = 2*int(ascending)-1
	for i in range(0, N-1):
		if s*A[i] > s*A[i+1]:
			flag=False
			break
	return flag

# print(check_sorted(range(10)))

def left_bound(A, key):
	left = -1
	right = len(A)
	while right - left > 1:
		middle = (left + right) // 2
		if A[middle] < key:
			left = middle
		else:
			right = middle
	return left

def right_bound(A, key):
	left = -1
	right = len(A)
	while right - left > 1:
		middle = (left + right) // 2
		if A[middle] <= key:
			left = middle
		else:
			right = middle
	return right

def fibo_dinamic(n):
	fib = [0, 1] + [0]*(n-1)
	for i in range(2, n+1):
		fib[i] = fib[i-1] + fib[i-2]
	return fib

def count_trajectories(N, allowed:list):
	K = [0, 1, int(allowed[2])] + [0] * (N-3)
	for i in range(3, N+1):
		if allowed[i]:
			K[i] = K[i-1] + K[i-2] + K[i-3]

def count_min_cost(N, price:list):
	C = [float("-inf"), price[1], price[1] + price[2]] + [0] * (N-2)
	for i in range(3, N+1):
		C[i] = price[i] + min(C[i-1], C[i-2])
	return C[N]

# двумерный массив

# A = [[0]*M for i in range(N)]


def lcs(A, B):
	# Наибольшая общая подпоследовательность (Longest common subsequence)
	F = [[0] * (len(B) + 1) for i in range(len(A) + 1)]
	for i in range(1, len(A) + 1):
		for j in range(1, len(B) + 1):
			if A[i-1] == B[j-1]:
				F[i][j] = 1 + F[i-1][j-1]
			else:
				F[i][j] = max(F[i-1][j], F[i][j-1])
	return F[-1][-1]


def gis(A):
	# Наибольшая возрастающая подпоследовательность (Greatest increasing subsequence)
	F = [0] * (len(A)+1)
	for i in range(1, len(A)+1):
		m = 0
		for j in range(0, i):
			if A[i] > A[j] and F[j] > m:
				m = F[j]
		F[i] = m + 1
	return F[len(A)]

def equal(A, B):
	if len(A) != len(B):
		return False
	for i in range(len(A)):
		if A[i] != B[i]:
			return False
	return True

def search_substring(s, sub):
	for i in range(0, len(s)-len(sub)):
		if equal(s[i:i+len(sub)], sub):
			print(i)

def kmp(s, sub):
	"""Алгоритм Кнута-Морриса-Пратта"""
	P = [0] * len(sub)
	j = 0
	i = 1
	while i < len(sub):
		if sub[i] == sub[j]:
			P[i] = j+1
			i += 1
			j += 1
		else:
			if j == 0:
				P[i] = 0
				i += 1
			else:
				j = P[j-1]
	k = 0
	l = 0
	while k < len(s):
		if s[k] == sub[l]:
			k += 1
			l += 1
			if l == len(sub):
				print("образ найден")
		else:
			if l == 0:
				k += 1
				if k == len(s):
					print("образ не найден")
			else:
				l = P[l-1]

# kmp("abcabeabcabcabd", "abcabd")


def levenstein(A, B):
	F = [[(i+j) if i*j==0 else 0 for j in range(len(B)+1)] for i in range(len(A)+1)]
	for i in range(1, len(A)+1):
		for j in range(1, len(B)+1):
			if A[i-1] == B[j-1]:
				F[i][j] = F[i-1][j-1]
			else:
				F[i][j] = 1+min(F[i-1][j], F[i][j-1])
	return F[len(A)][len(B)]


def shell_sort(A):
	def gap_insertion_sort(A, start, gap):
		for i in range(start+gap, len(A), gap):

			current_value = A[i]
			pos = i

			while pos >= gap and A[pos-gap] > current_value:
				A[pos] = A[pos-gap]
				pos = pos - gap

			A[pos] = current_value

	sub_list_count = len(A)//2
	while sub_list_count > 0:
		for start_position in range(sub_list_count):
			gap_insertion_sort(A, start_position, sub_list_count)

		sub_list_count = sub_list_count // 2


class BinHeap:
	def __init__(self):
		self.heaplist = []
		self.heapsize = 0

	def left(self, i):
		return i * 2 + 1

	def right(self, i):
		return i * 2 + 2

	def heapify(self, i):
		l = self.left(i)
		r = self.right(i)
		if l <= self.heapsize and self.heaplist[l] > self.heaplist[i]:
			largest = l
		else:
			largest = i
		if r <= self.heapsize and self.heaplist[r] > self.heaplist[largest]:
			largest = r
		if largest != i:
			tmp = self.heaplist[i]
			self.heaplist[i] = self.heaplist[largest]
			self.heaplist[largest] = tmp
			self.heapify(largest)

	def buildHeap(self, list):
		self.heaplist = list
		self.heapsize = len(list) - 1
		for i in range(len(list) // 2, -1, -1):
			# print(i)
			self.heapify(i)

	def heapSort(self):
		pass

	def extractMax(self):
		pass

	def getHeap(self):
		return self.heaplist

heap = BinHeap()
heap.buildHeap([0, 0, 9, 5, 23, 0, 0, 2, 2, 1, 4, 0, 12, -1, 0])
print(heap.getHeap())

"Строка master after"