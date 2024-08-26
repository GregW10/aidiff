import os
import sys


def main():
	m = 10
	if len(sys.argv) > 1:
		m = int(sys.argv[1])
	files = [f for f in os.listdir() if os.path.isfile(f)]
	nums = {}
	for fname in files:
		i = 0
		num = ""
		for char in fname:
			if char.isdigit():
				for d in fname[i:]:
					if not d.isdigit():
						break
					num += d
				nums[fname] = int(num)
				i += 1
				break
			i += 1
	for (key, val) in zip(nums.keys(), nums.values()):
		if val % m != 0:
			os.remove(key)
			print(f"Removed \"{key}\".")


if __name__ == "__main__":
	main()
