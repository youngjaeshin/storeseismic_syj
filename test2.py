import urllib.request

# 실제 BIP-39 단어 리스트 불러오기
url = "https://raw.githubusercontent.com/bitcoin/bips/master/bip-0039/english.txt"
wordlist = urllib.request.urlopen(url).read().decode().splitlines()

# 예시 인덱스
indexes = [582, 204, 1359, 985, 99, 660, 71, 575, 376, 1097, 1587, 1173]
mnemonic = [wordlist[i] for i in indexes]
print(" ".join(mnemonic))
