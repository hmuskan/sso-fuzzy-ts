from rouge import FilesRouge
files_rouge = FilesRouge()
scores = files_rouge.get_scores('6.summarized/001.txt', '3.reference/001.txt', avg=True)
print(scores)
# or
# scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)

# path_out2 = './6.summarized/'
# with open('./3.reference/001.txt') as f:
#     num_lines_ref = sum(1 for _ in f)
#
# with open(path_out2 + '001.txt') as f:
#     num_lines_sys = sum(1 for _ in f)
# print(num_lines_sys)
# print(num_lines_ref)
#
# while num_lines_sys > num_lines_ref:
#     print(num_lines_sys - num_lines_ref)
#     fd = open(path_out2 + '001.txt', "r")
#     d = fd.read()
#     fd.close()
#     m = d.split("\n")
#     s = "\n".join(m[:-1])
#     fd = open(path_out2 + '001.txt', "w+")
#     for i in range(len(s)):
#         fd.write(s[i])
#     fd.close()
#     with open(path_out2 + '001.txt') as f:
#         num_lines_sys = sum(1 for _ in f)