
import sys

def siamese(string1, string2):
    string1 = sorted(list(string1))
    string2 = sorted(list(string2))
    string1.remove(' ')
    string2.remove(' ')
    string1 = "".join(string1)
    string2 = "".join(string2)
    if string1 == string2:
        return True
    return False
        


if __name__ == '__main__':
    string1 = sys.argv[1]
    string2 = sys.argv[2]
    if siamese(string1, string2):
        print("Siamese Strings")
    else:
        print("Not Siamese Strings")
